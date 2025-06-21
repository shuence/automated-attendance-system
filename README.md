# Automated Facial Attendance System

An AI-driven facial recognition system for automating classroom attendance using Streamlit and DeepFace.

## Features

- **Student Registration**: Register students with face images, email addresses, and subject enrollment
- **Teacher Dashboard**: Select subject and period, upload or capture classroom images
- **Facial Recognition**: Automatically identify students present in class images
- **Attendance Records**: Store attendance in SQLite database and Excel sheets
- **ESP32-CAM Integration**: Capture classroom images via HTTP request
- **Email Integration**: Store student emails for notifications (future enhancement)
- **Robust Error Handling**: Improved error handling for dependency issues and runtime errors

## Project Structure

``` bash
project/
├── app.py                 # Main Streamlit application
├── setup.py               # Environment setup script
├── test_app.py            # Test script to verify setup
├── run_app.bat            # Windows script to run the app
├── run_app.sh             # Linux/macOS script to run the app
├── requirements.txt       # Python dependencies
├── db/                    # Database directory
│   └── attendance.db      # SQLite database
├── faces/                 # Student face images
│   └── ...
├── excel_exports/         # Exported Excel reports
│   └── ...
└── utils/                 # Utility modules
    ├── deepface_utils.py  # Facial recognition utilities
    ├── db_utils.py        # Database utilities
    └── migrate_db.py      # Database migration script
``` bash

## Installation

### Automatic Setup (Recommended)

1. Clone the repository:
   ``` bash
   git clone https://github.com/yourusername/facial-attendance-system.git
   cd facial-attendance-system
   ```

2. Run the setup script:
   ``` bash
   python setup.py
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - MacOS/Linux: `source venv/bin/activate`

### Quick Start (Windows)

```
run_app.bat
```

### Quick Start (Linux/macOS)

```
chmod +x run_app.sh
./run_app.sh
```

### Manual Setup

1. Clone the repository:
   ``` bash
   git clone https://github.com/yourusername/facial-attendance-system.git
   cd facial-attendance-system
   ```

2. Create a virtual environment:
   ``` bash
   python -m venv venv
   ``` bash

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - MacOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ``` bash
   pip install -r requirements.txt
   ``` bash

## Testing

To verify that all dependencies are installed correctly and the application is ready to run:

``` bash
python test_app.py
```

## Usage

1. Run the Streamlit application:
   ``` bash
   streamlit run app.py
   ``` bash

2. Access the web interface at `http://localhost:8501`

3. Use the sidebar to navigate between:
   - Teacher Dashboard
   - Student Registration
   - Attendance Reports

### Student Registration

1. Fill the registration form with:
   - Full name
   - Roll number (e.g., EC3201)
   - Email address (for future notifications)
   - Upload a clear face image
   - Select enrolled subjects

2. Submit the form to register the student

### Teacher Dashboard

1. Select the subject and period
2. Choose an image source:
   - Upload an image from your device
   - Capture from an ESP32-CAM (requires setup)
3. Click "Process Attendance" to detect students and mark attendance
4. View the attendance summary

### Attendance Reports

1. Select a subject
2. Choose a date
3. View the attendance report
4. Download as Excel file if needed
5. Option to send email notifications to absent students (feature in development)

## ESP32-CAM Setup

1. Upload the ESP32-CAM code (available in the `esp32_cam` branch)
2. Connect the ESP32-CAM to the same network as your computer
3. Note the IP address of your ESP32-CAM
4. Enter the URL `http://[ESP32-CAM-IP]/capture` in the Teacher Dashboard

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI**: DeepFace with TensorFlow/Keras
- **Database**: SQLite
- **Reporting**: Pandas, OpenPyXL
- **Hardware**: ESP32-CAM (optional)

## Database Migration

If you've previously installed the application without the email field, you can run:

``` bash
python -m utils.migrate_db
```

This will add the email column to your existing database.

## Troubleshooting

### DeepFace Installation Issues

If you encounter issues with DeepFace or TensorFlow:

1. Make sure you have compatible versions:
   ``` bash
   pip install tensorflow==2.12.0 keras==2.12.0
   pip install deepface==0.0.79
   ```

2. If you see "cannot import name 'LocallyConnected2D'" error:
   - This is a compatibility issue between newer TensorFlow/Keras versions and DeepFace
   - The fixed requirements.txt in this repo should resolve this issue

3. For other DeepFace errors:
   - Check the app.log and deepface.log files
   - The application is designed to handle most DeepFace errors gracefully

### Database Issues

If you encounter database errors:

1. Check if the database file exists:
   ``` bash
   python -c "import os; print(os.path.exists('db/attendance.db'))"
   ```

2. If not, try initializing it manually:
   ``` bash
   python -c "from utils.db_utils import init_db; init_db()"
   ```

3. If there are issues with an existing database:
   ``` bash
   # Backup the database
   cp db/attendance.db db/attendance.db.backup
   
   # Remove and recreate
   rm db/attendance.db
   python -c "from utils.db_utils import init_db; init_db()"
   ```

### Image Processing Issues

If face detection doesn't work correctly:

1. Make sure the images are clear and faces are visible
2. Try different threshold values in utils/deepface_utils.py
3. Check if the required models are downloaded (should happen automatically)

## Recent Enhancements

- Added email field for student registration
- Included email notifications feature (coming soon)
- Fixed compatibility issues with TensorFlow and DeepFace
- Improved setup process with automatic dependency management
- Enhanced Excel reports with email information
- Added robust error handling throughout the application
- Created test script to verify installation and dependencies
- Added detailed logging for troubleshooting

## Future Enhancements

- Active email notifications for absent students
- Role-based login (Admin/Teacher)
- Dashboard view for student analytics
- Live video attendance using webcam
- Notifications to students on absenteeism
- Multidepartment, multiyear scalability
- Face data training from webcam or phone
- Integration with Google Sheets or Firebase

## License

MIT 