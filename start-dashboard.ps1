# PowerShell script to start both FastAPI backend and React frontend for the dashboard

Start-Process powershell -ArgumentList "cd benchmark-dashboard\backend; uvicorn main:app --reload"
Start-Process powershell -ArgumentList "cd dashboard; npm run dev"
