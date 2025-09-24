# batch_grader.py
import asyncio
from team import team  # Import your existing team setup
from async_batch_grader import AsyncBatchGrader  # The code I provided

async def main():
    grader = AsyncBatchGrader(
        team=team,
        batch_size=5,
        max_concurrent=5,
        retry_attempts=3
    )
    
    summary = await grader.process_excel_file(
        "data/student_response.xlsx", 
        "data/graded_responses.json"
    )
    
    print(f"Processed {summary['total_students']} students")
    print(f"Success rate: {summary['success_rate_percent']}%")

if __name__ == "__main__":
    asyncio.run(main())