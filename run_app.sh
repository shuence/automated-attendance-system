#!/bin/bash

echo "Starting Facial Attendance System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    python3 setup.py
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Start Streamlit application
echo "Starting Streamlit application..."
streamlit run app.py 