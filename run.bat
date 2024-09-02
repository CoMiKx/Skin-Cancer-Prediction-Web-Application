@echo offrun
REM Check if a virtual environment already exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install the dependencies from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

REM Run the Python script
echo Running main.py...
python main.py

REM Keep the window open after execution
pause
