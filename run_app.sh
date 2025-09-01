#!/bin/bash

echo "🏥 Starting AI Radiology System..."

# Start API server in background
echo "🚀 Starting API server..."
python api/main.py &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit
echo "🌐 Starting Streamlit UI..."
streamlit run streamlit_app.py

# Cleanup on exit
trap "kill $API_PID" EXIT