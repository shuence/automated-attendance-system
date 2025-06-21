import os
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed and compatible"""
    dependencies = [
        'streamlit',
        'pandas',
        'PIL.Image',
        'numpy',
        'requests',
        'tensorflow',
        'keras',
        'openpyxl',
        'sqlite3'
    ]
    
    print("Checking dependencies...")
    all_ok = True
    
    for module_name in dependencies:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown version')
            print(f"✅ {module_name}: {version}")
        except ImportError:
            print(f"❌ {module_name}: Not installed")
            all_ok = False
    
    # Check DeepFace separately
    try:
        from deepface import DeepFace
        version = getattr(DeepFace, '__version__', 'Unknown version')
        print(f"✅ deepface: {version}")
    except ImportError as e:
        print(f"❌ deepface: Not installed or error - {str(e)}")
        all_ok = False
    
    return all_ok

def check_folders():
    """Check if all required folders exist, create if not"""
    folders = ['db', 'faces', 'excel_exports']
    
    print("\nChecking folders...")
    for folder in folders:
        if os.path.exists(folder):
            print(f"✅ {folder}/: Exists")
        else:
            print(f"⚠️ {folder}/: Creating folder")
            os.makedirs(folder, exist_ok=True)

def check_database():
    """Check if database can be initialized"""
    print("\nChecking database...")
    try:
        from utils.db_utils import init_db
        init_db()
        print("✅ Database: Successfully initialized")
        return True
    except Exception as e:
        print(f"❌ Database: Error initializing - {str(e)}")
        return False

def run_tests():
    """Run basic functionality tests"""
    print("\n=== Running Facial Attendance System Tests ===\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check folders
    check_folders()
    
    # Check database
    db_ok = check_database()
    
    # Overall results
    print("\n=== Test Results ===")
    if deps_ok and db_ok:
        print("✅ All tests passed! The application should work correctly.")
        print("\nYou can run the application with:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the application.")

if __name__ == "__main__":
    run_tests() 