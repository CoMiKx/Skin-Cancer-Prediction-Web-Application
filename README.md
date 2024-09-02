# Skin-Cancer-Prediction-Web-Application

Welcome to the **Skin Cancer Prediction Web Application**! This project helps you get up and running with a local instance of our prediction model. Grab your coffee and let's dive in—because setting up a project is much better with caffeine!

## 🚀 Getting Started

We’ve made it super easy to set up this project with a simple batch file. Just follow these steps:

### Option 1: Using `run.bat` (The Developer’s Choice ☕)

1. **Double-click `run.bat`:**  
   This script will do all the heavy lifting:
   - It creates a virtual environment (if it doesn’t already exist).
   - Installs all the required dependencies from `requirements.txt`.
   - Runs the project for you.

   ⚠️ Heads up! The setup might take a bit of time depending on your machine. So, feel free to grab a cup of coffee and relax while the magic happens!

2. **Access the Web App:**  
   Once everything is set up, the web application will be available at [http://localhost:8000](http://localhost:8000).

   Enjoy exploring the world of skin cancer detection!

### Option 2: Using `run.exe` (The “No Setup Needed” Way)

If you’re in a hurry or don’t feel like dealing with setting up the environment, just use `run.exe`. It takes care of everything, and the web application will be live at [http://localhost:8000](http://localhost:8000) in no time.

## 🛠 How `run.bat` Works

Here’s what happens behind the scenes when you run the batch file:

```bat
@echo off
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
```
### Key Steps:
- **Create a Virtual Environment:** Ensures all dependencies are isolated to this project.
- **Install Dependencies:** Installs everything needed from `requirements.txt`.
- **Run the Project:** Starts the web server and launches the application.

## 🔗 Accessing the Web Application
After setup, you can access the application at: 
```bat
http://localhost:8000
```
