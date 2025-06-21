@echo off
echo Starting Facial Attendance System...

IF NOT EXIST venv (
    echo Virtual environment not found. Running setup...
    python setup.py
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Checking database...
IF NOT EXIST db\attendance.db (
    echo Database not found. Initializing...
    python -c "from utils.db_utils import init_db; init_db()"
    echo.
) ELSE (
    echo Database found.
    echo If you encounter database errors, run with --reset-db to rebuild the database.
    echo.
)

if "%1"=="--reset-db" (
    echo Resetting database...
    if exist db\attendance.db.backup del db\attendance.db.backup
    if exist db\attendance.db rename db\attendance.db attendance.db.backup
    python -c "from utils.db_utils import init_db; init_db()"
    echo Database reset complete.
    echo.
)

echo Starting Streamlit application...
streamlit run app.py

pause 