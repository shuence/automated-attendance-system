# Facial Attendance System

A comprehensive facial recognition-based attendance system for educational institutions, designed specifically for ENTC TYB students.

![Attendance System](https://img.shields.io/badge/Facial-Attendance-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-red)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79-orange)

## Overview

This application uses advanced facial recognition technology to automate the attendance tracking process in educational institutions. Designed primarily for ENTC TYB students, it offers features like real-time attendance marking, detailed reports generation, analytics, and attendance management.

## Features

- **Student Registration**: Register students with facial recognition data
- **Attendance Marking**: Take attendance using facial recognition from classroom images
- **Dashboard**: Comprehensive teacher dashboard with attendance statistics
- **Reports**: Generate detailed attendance reports by student, class, and subject
- **Export**: Export attendance data to Excel format
- **Analytics**: Analyze attendance patterns and recognition statistics

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam for live attendance (optional)
- Modern web browser

## Installation & Setup

### Windows

1. Clone the repository
2. Run the setup script:
   ```
   run_app.bat
   ```

### Linux/macOS

1. Clone the repository
2. Make the setup script executable:
   ```
   chmod +x run_app.sh
   ```
3. Run the setup script:
   ```
   ./run_app.sh
   ```

### Manual Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application with Streamlit interface
- `utils/`: Utility functions
  - `db_utils.py`: Database operations
  - `deepface_utils.py`: Facial recognition functions
  - `migrate_db.py`: Database migration utilities
- `db/`: SQLite database directory
- `faces/`: Directory for storing student face images
- `excel_exports/`: Directory for exported Excel reports
- `attendance/`: Attendance data and processing

## Technology Stack

- **Streamlit**: Web application framework
- **DeepFace**: Facial recognition and analysis
- **SQLite**: Database for storing student and attendance data
- **Pandas**: Data analysis and manipulation
- **Plotly**: Interactive visualizations
- **OpenCV**: Image processing
- **TensorFlow/Keras**: Backend for facial recognition

## Usage Guide

1. **Student Registration**:
   - Navigate to "Student Registration"
   - Enter student details and upload a clear face photo
   - Select subjects for the student

2. **Taking Attendance**:
   - Navigate to "Take Attendance"
   - Upload a classroom image
   - Select subject and period
   - The system will automatically recognize students and mark attendance

3. **Viewing Reports**:
   - Navigate to "Attendance Reports" for individual student reports
   - Navigate to "Class-wise Reports" for class-level reports
   - Filter by date, subject, or student as needed

4. **Exporting Data**:
   - Use the export buttons in the reports sections to download Excel reports

## Database Reset

If you need to reset the database:

- Windows: `run_app.bat --reset-db`
- Linux/macOS: Manually delete `db/attendance.db` and restart the application

## Troubleshooting

- **Recognition Issues**: Ensure good lighting and clear face visibility in photos
- **Database Errors**: Try resetting the database with the --reset-db flag
- **Missing Dependencies**: Re-run the setup script to install all required packages
- **Log Files**: Check app.log and deepface.log for error details

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- DeepFace for facial recognition capabilities
- Streamlit for the interactive web interface
- TensorFlow and Keras for the underlying ML models