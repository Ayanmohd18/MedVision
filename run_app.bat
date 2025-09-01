@echo off
echo Starting AI Radiology System...
echo.

echo Starting API server...
start "API Server" cmd /k "python api/main.py"

echo Waiting for API to start...
timeout /t 3 /nobreak > nul

echo Starting Streamlit UI...
streamlit run streamlit_app.py

pause