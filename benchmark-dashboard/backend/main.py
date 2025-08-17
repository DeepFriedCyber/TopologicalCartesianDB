from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import glob
import pandas as pd

app = FastAPI()

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../comparison_results'))

@app.get("/api/latest-results")
def get_latest_results():
    # Find latest JSON or CSV file in the results directory
    files = glob.glob(os.path.join(RESULTS_DIR, '*.json')) + glob.glob(os.path.join(RESULTS_DIR, '*.csv'))
    if not files:
        raise HTTPException(status_code=404, detail="No results found.")
    latest_file = max(files, key=os.path.getmtime)
    ext = os.path.splitext(latest_file)[1].lower()
    if ext == '.json':
        df = pd.read_json(latest_file)
    elif ext == '.csv':
        df = pd.read_csv(latest_file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "filename": os.path.basename(latest_file)}
