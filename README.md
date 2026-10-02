# CoastGuard AI
AI-powered coastal plastic pollution monitoring (InnovateX, Techfest IIT Bombay).

## Run the backend
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload

API docs: http://127.0.0.1:8000/docs