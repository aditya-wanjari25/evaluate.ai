# evaluate.ai

A multi-agent system that evaluates student assignments. 💯

Features:
* 🤖 Specialzed Agents for each rubric: Easily modify prompts and agent behavior to match your specific grading requirements
* ✅ Provide Score and Tips for Improvement: Generates numerical scores plus detailed feedback for each evaluation criteria
* 📑 RAG Integrated with Postgres: Add reference material to validate facts!
* ䷶ Async + Batch Processing for Bulk Uploads: Process hundreds of assignments simultaneously with concurrent execution


## Quick Start
1. Clone the repo
2. Run the app ```docker-compose up --build```
3. For batch grading ```docker-compose run grader python batch_grader.py```
