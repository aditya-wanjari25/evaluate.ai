# evaluate.ai

A multi-agent system that evaluates student assignments. 💯

Features:
* 🤖 Specialzed Agents for each rubric: Easily modify prompts and agent behavior to match your specific grading requirements
* ✅ Provide Score and Tips for Improvement: Generates numerical scores plus detailed feedback for each evaluation criteria
* 📑 RAG Integrated with Postgres: Add reference material to validate facts!
* ䷶ Async + Batch Processing for Bulk Uploads: Process hundreds of assignments simultaneously with concurrent execution


## Quick Set Up
Install Requirements
```
pip install -r requirements.txt
```

Setup Postgres DB for RAG
```
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

Run team.py to evaluate a single response.
```
python team.py
```

Run batch_grader.py to evaluate bulk responses.
```
python batch_grader.py
```
