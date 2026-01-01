# ML Ensemble DAG Builder

## Project Structure
- `backend/`: FastAPI backend (Python)
- `frontend/`: Next.js frontend (TypeScript/React)

## How to Run

### Backend
The backend runs on port **8001**.
```bash
python backend/main.py
```
*Note: Ensure you have the required python packages installed (fastapi, uvicorn, pandas, etc).*

### Frontend
The frontend runs on port **3000** (default Next.js port).
```bash
cd frontend
npm run dev
```

## Configuration
- The frontend is configured to communicate with the backend at `http://localhost:8001`.
- This is defined in `frontend/utils/api.ts`.
