"""
Test script to troubleshoot attendance saving and Excel export issues
"""
import os
import datetime
import pandas as pd

from utils.db_utils import (
    get_connection,
    get_subjects,
    get_all_students,
    mark_attendance,
    get_attendance_report
)

def test_attendance_marking():
    """Test marking attendance directly"""
    print("--- Testing Attendance Marking ---")
    
    # Get all students
    students = get_all_students()
    if not students:
        print("No students found in database!")
        return
    
    print(f"Found {len(students)} students")
    
    # Get all subjects
    subjects = get_subjects()
    if not subjects:
        print("No subjects found in database!")
        return
    
    print(f"Found {len(subjects)} subjects")
    
    # Use the first student and first subject for testing
    student = students[0]
    subject = subjects[0]
    
    student_id = student["id"]
    subject_id = subject[0]
    
    print(f"Testing with: Student ID {student_id} ({student['name']}) - Subject ID {subject_id} ({subject[1]})")
    
    # Mark attendance for today
    today = datetime.date.today().strftime("%Y-%m-%d")
    period = "10:15 - 11:15"
    
    print(f"Marking attendance for date: {today}, period: {period}")
    success = mark_attendance(student_id, subject_id, today, period)
    
    if success:
        print("Attendance was successfully marked!")
    else:
        print("Failed to mark attendance!")
    
    # Verify by querying the database directly
    print("\nVerifying attendance record:")
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM attendance 
    WHERE student_id = ? AND subject_id = ? AND date = ? AND period = ?
    ''', (student_id, subject_id, today, period))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        print(f"Attendance record found: {dict(result)}")
    else:
        print("No attendance record found in the database!")
    
    # Try to export to Excel
    print("\n--- Testing Excel Export ---")
    try:
        report = get_attendance_report(subject_id, today)
        if report:
            print(f"Found attendance records for export: {len(report)}")
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {"Roll No": row[1], "Name": row[2], "Email": row[3], "Status": row[4]}
                for row in report
            ])
            
            # Specify the test file path in the excel_exports directory
            excel_path = os.path.join("excel_exports", "test_export.xlsx")
            
            print(f"Exporting to Excel: {excel_path}")
            df.to_excel(excel_path, index=False)
            
            if os.path.exists(excel_path):
                file_size = os.path.getsize(excel_path)
                print(f"Excel file successfully created! Size: {file_size} bytes")
            else:
                print("Failed to create Excel file!")
        else:
            print("No data found for export")
    except Exception as e:
        print(f"Error during Excel export: {str(e)}")

if __name__ == "__main__":
    test_attendance_marking() 