@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay Python trong .venv. Xem docs\07-streamlit-demo.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
