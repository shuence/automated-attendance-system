import os
import subprocess
import sys

def setup_environment():
    """
    Set up the project environment with compatible package versions
    """
    print("Setting up the Facial Attendance System environment...")
    
    # Check if virtual environment exists, create if not
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine the pip path
    if os.name == "nt":  # Windows
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:  # macOS/Linux
        pip_path = os.path.join("venv", "bin", "pip")
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    
    # Install requirements with specific versions
    print("Installing dependencies with compatible versions...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    print("\nSetup complete! Activate your virtual environment with:")
    if os.name == "nt":  # Windows
        print("venv\\Scripts\\activate")
    else:  # macOS/Linux
        print("source venv/bin/activate")
    
    print("\nThen run the application with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    setup_environment() 