import asyncio
import pandas as pd
import time
from typing import List, Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    total_students: int = 0
    completed: int = 0
    failed: int = 0
    start_time: float = 0
    
    def update_progress(self):
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total_students - self.completed) / rate if rate > 0 else 0
        
        logger.info(f"Progress: {self.completed}/{self.total_students} "
                   f"({self.completed/self.total_students*100:.1f}%) "
                   f"Rate: {rate:.1f} students/sec "
                   f"ETA: {eta:.1f}s")

class AsyncBatchGrader:
    def __init__(self, team, batch_size=10, max_concurrent=15, retry_attempts=3):
        self.team = team
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.stats = ProcessingStats()
        
    async def process_excel_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Main function to process Excel file with student responses
        """
        logger.info(f"Starting batch processing: {input_file}")
        
        # Load student data
        try:
            df = pd.read_excel(input_file)
            logger.info(f"Loaded {len(df)} students from Excel file")
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
        
        # Initialize stats
        self.stats.total_students = len(df)
        self.stats.start_time = time.time()
        
        # Process all students
        results = await self._process_all_students(df)
        
        # Save results as JSON
        await self._save_results_json(results, output_file)
        
        # Return summary
        return self._generate_summary()
    
    async def _process_all_students(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process all students in batches with async execution
        """
        results = []
        
        # Process in batches
        for batch_start in range(0, len(df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//self.batch_size + 1}: "
                       f"students {batch_start+1}-{batch_end}")
            
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Create tasks for this batch
            batch_tasks = [
                self._process_single_student_with_semaphore(semaphore, row)
                for _, row in batch_df.iterrows()
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    self.stats.failed += 1
                    results.append(self._create_error_result("Unknown", str(result)))
                else:
                    results.append(result)
                    if 'error' in result:
                        self.stats.failed += 1
                    else:
                        self.stats.completed += 1
            
            # Update progress
            self.stats.update_progress()
            
            # Small delay between batches to prevent overwhelming the API
            if batch_end < len(df):
                await asyncio.sleep(0.5)
        
        return results
    
    async def _process_single_student_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                                   student_row: pd.Series) -> Dict[str, Any]:
        """
        Process a single student with semaphore control
        """
        async with semaphore:
            return await self._process_single_student_with_retry(student_row)
    
    async def _process_single_student_with_retry(self, student_row: pd.Series) -> Dict[str, Any]:
        """
        Process a single student with retry logic
        """
        student_id = student_row.get('student_id', 'unknown')
        
        for attempt in range(self.retry_attempts):
            try:
                result = await self._process_single_student(student_row)
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for student {student_id}: {e}")
                
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Final attempt failed
                    logger.error(f"All attempts failed for student {student_id}")
                    return self._create_error_result(student_id, str(e))
    
    async def _save_results_json(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file"""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Change extension to .json if it's .xlsx
            if output_file.endswith('.xlsx'):
                output_file = output_file.replace('.xlsx', '.json')
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
            
            # Also create a summary CSV for easy viewing
            csv_file = output_file.replace('.json', '_summary.csv')
            summary_data = []
            for result in results:
                if 'error' not in result:
                    summary_data.append({
                        'student_id': result.get('ID'),
                        'total_score': result.get('total_score'),
                        'industry_analysis_score': result.get('industry_analysis_score'),
                        'comparison_score': result.get('comparison_score'),
                        'rag_score': result.get('rag_score'),
                        'presentation_score': result.get('presentation_score')
                    })
                else:
                    summary_data.append({
                        'student_id': result.get('ID'),
                        'total_score': 0,
                        'error': result.get('error')
                    })
            
            if summary_data:
                pd.DataFrame(summary_data).to_csv(csv_file, index=False)
                logger.info(f"Summary CSV saved to {csv_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    async def _process_single_student(self, student_row: pd.Series) -> Dict[str, Any]:
        """
        Process a single student's response using the team
        """
        student_id = student_row.get('ID', 'unknown')
        response_text = student_row.get('Response', '')
        
        if not response_text.strip():
            return self._create_error_result(student_id, "Empty response")
        
        # Run the team processing in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            team_response = await loop.run_in_executor(
                self.executor,
                lambda: self.team.run(input=response_text)
            )
            
            # Extract the structured response
            response_content = team_response.content
            
            # Create result dictionary (no cleaning needed for JSON)
            result = {
                'student_id': student_id,
                'processing_time': time.time(),
                'industry_analysis_score': response_content.industry_analysis_score,
                'industry_analysis_feedback': response_content.industry_analysis_feedback,
                'comparison_score': response_content.comparison_score,
                'comparison_feedback': response_content.comparison_feedback,
                'rag_score': response_content.rag_score,
                'rag_feedback': response_content.rag_feedback,
                'presentation_score': response_content.presentation_score,
                'presentation_feedback': response_content.presentation_feedback,
                'total_score': (
                    response_content.industry_analysis_score +
                    response_content.comparison_score +
                    response_content.rag_score +
                    response_content.presentation_score
                )
            }
            
            logger.debug(f"Successfully processed student {student_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing student {student_id}: {e}")
            raise
    
    def _create_error_result(self, student_id: str, error_message: str) -> Dict[str, Any]:
        """Create a result object for failed processing"""
        return {
            'student_id': student_id,
            'error': error_message,
            'processing_time': time.time(),
            'industry_analysis_score': 0,
            'industry_analysis_feedback': 'Processing failed',
            'comparison_score': 0,
            'comparison_feedback': 'Processing failed',
            'rag_score': 0,
            'rag_feedback': 'Processing failed',
            'presentation_score': 0,
            'presentation_feedback': 'Processing failed',
            'total_score': 0
        }
    
    def _clean_text_for_excel(self, text: Any) -> str:
        """Clean text to remove characters that cause Excel issues"""
        if text is None:
            return ""
        
        text = str(text)
        
        # Remove or replace problematic characters
        # Remove null bytes and other control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        # Replace smart quotes and other problematic Unicode
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Truncate if too long (Excel cell limit is ~32767 characters)
        if len(text) > 32000:
            text = text[:32000] + "... [truncated]"
            
        return text
    
    async def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to Excel file"""
        try:
            # Clean all text data before creating DataFrame
            cleaned_results = []
            for result in results:
                cleaned_result = {}
                for key, value in result.items():
                    if isinstance(value, str) or key.endswith('_feedback'):
                        cleaned_result[key] = self._clean_text_for_excel(value)
                    else:
                        cleaned_result[key] = value
                cleaned_results.append(cleaned_result)
            
            df_results = pd.DataFrame(cleaned_results)
            
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to Excel with additional error handling
            try:
                df_results.to_excel(output_file, index=False, engine='openpyxl')
                logger.info(f"Results saved to {output_file}")
            except Exception as excel_error:
                # If Excel save fails, save as CSV instead
                csv_file = output_file.replace('.xlsx', '.csv')
                df_results.to_csv(csv_file, index=False)
                logger.warning(f"Excel save failed, saved as CSV instead: {csv_file}")
                logger.error(f"Excel error: {excel_error}")
            
            # Also save as JSON for backup
            json_file = output_file.replace('.xlsx', '_backup.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate processing summary"""
        total_time = time.time() - self.stats.start_time
        success_rate = (self.stats.completed / self.stats.total_students * 100) if self.stats.total_students > 0 else 0
        throughput = self.stats.total_students / total_time if total_time > 0 else 0
        
        summary = {
            'total_students': self.stats.total_students,
            'completed': self.stats.completed,
            'failed': self.stats.failed,
            'success_rate_percent': round(success_rate, 2),
            'total_time_seconds': round(total_time, 2),
            'throughput_students_per_minute': round(throughput * 60, 2),
            'average_time_per_student': round(total_time / self.stats.total_students, 2) if self.stats.total_students > 0 else 0
        }
        
        logger.info("Processing Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return summary
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

# Usage example
async def main():
    """
    Example usage of the AsyncBatchGrader
    """
    # Import your existing team setup
    from team import team  # Your existing team setup
    
    # Create the batch grader
    grader = AsyncBatchGrader(
        team=team,
        batch_size=5,          # Process 15 students per batch
        max_concurrent=5,      # Max 10 concurrent operations
        retry_attempts=3        # Retry failed attempts 3 times
    )
    
    try:
        # Process the Excel file
        summary = await grader.process_excel_file(
            input_file="data/student_responses.xlsx",
            output_file="data/graded_responses.json"
        )
        
        print("\n" + "="*50)
        print("BATCH PROCESSING COMPLETED")
        print("="*50)
        print(f"Total students processed: {summary['total_students']}")
        print(f"Success rate: {summary['success_rate_percent']}%")
        print(f"Total time: {summary['total_time_seconds']} seconds")
        print(f"Throughput: {summary['throughput_students_per_minute']} students/minute")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise
    finally:
        grader.__exit__(None, None, None)

# Run the async batch processing
if __name__ == "__main__":
    asyncio.run(main())