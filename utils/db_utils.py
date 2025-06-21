import os
import sqlite3
import datetime

# Database configuration
DB_PATH = os.path.join("db", "attendance.db")

def get_connection():
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable row factory for column name access
    return conn

def init_db():
    """Initialize the database with required tables if they don't exist"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        email TEXT,
        department TEXT DEFAULT 'ENTC',
        year TEXT DEFAULT 'TY',
        division TEXT DEFAULT 'B',
        image_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create subjects table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL
    )
    ''')
    
    # Create student_subjects table (many-to-many relationship)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student_subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_id INTEGER NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
        FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE,
        UNIQUE (student_id, subject_id)
    )
    ''')
    
    # Create attendance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        period TEXT NOT NULL,
        status TEXT DEFAULT 'present',
        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
        FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE,
        UNIQUE (student_id, subject_id, date, period)
    )
    ''')
    
    # Insert default subjects if they don't exist
    subjects = [
        ("AWP", "Antenna and Wave Propagation"),
        ("DC", "Digital Communication"),
        ("CN", "Computer Networks"),
        ("M&M", "Microprocessors & Microcontrollers"),
        ("ESD", "Embedded System Design"),
        ("MML", "Microprocessors & Microcontrollers Lab"),
        ("DCL", "Digital Communication Lab"),
        ("MP", "Mini Project"),
        ("SEM", "Seminar"),
    ]
    
    for code, name in subjects:
        cursor.execute('''
        INSERT OR IGNORE INTO subjects (code, name) VALUES (?, ?)
        ''', (code, name))
    
    conn.commit()
    conn.close()

def register_student(roll_no, name, department, year, division, image_path, subject_ids, email=None):
    """Register a new student in the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Insert student
        cursor.execute('''
        INSERT INTO students (roll_no, name, email, department, year, division, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (roll_no, name, email, department, year, division, image_path))
        
        # Get the ID of the newly inserted student
        student_id = cursor.lastrowid
        
        # Associate student with subjects
        for subject_id in subject_ids:
            cursor.execute('''
            INSERT INTO student_subjects (student_id, subject_id)
            VALUES (?, ?)
            ''', (student_id, subject_id))
        
        conn.commit()
        return student_id
    except sqlite3.IntegrityError:
        # Roll number already exists
        conn.rollback()
        return None
    finally:
        conn.close()

def get_all_students():
    """Get all students from the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, roll_no, name, email, department, year, division, image_path
    FROM students
    ORDER BY roll_no
    ''')
    
    # Convert to list of dictionaries
    students = []
    for row in cursor.fetchall():
        students.append({
            "id": row["id"],
            "roll_no": row["roll_no"],
            "name": row["name"],
            "email": row["email"],
            "department": row["department"],
            "year": row["year"],
            "division": row["division"],
            "image_path": row["image_path"]
        })
    
    conn.close()
    return students

def get_students_by_subject(subject_id):
    """Get all students enrolled in a specific subject"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT s.id, s.roll_no, s.name, s.email, s.image_path
    FROM students s
    JOIN student_subjects ss ON s.id = ss.student_id
    WHERE ss.subject_id = ?
    ORDER BY s.roll_no
    ''', (subject_id,))
    
    # Convert to list of dictionaries
    students = []
    for row in cursor.fetchall():
        students.append({
            "id": row["id"],
            "roll_no": row["roll_no"],
            "name": row["name"],
            "email": row["email"],
            "image_path": row["image_path"]
        })
    
    conn.close()
    return students

def get_subjects():
    """Get all subjects from the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, code, name FROM subjects ORDER BY name
    ''')
    
    subjects = cursor.fetchall()
    conn.close()
    
    return subjects

def mark_attendance(student_id, subject_id, date, period, status="present"):
    """Mark attendance for a student"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO attendance (student_id, subject_id, date, period, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (student_id, subject_id, date, period, status))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_attendance_report(subject_id, date):
    """Get attendance report for a specific subject and date"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        s.id, 
        s.roll_no, 
        s.name,
        s.email,
        COALESCE(a.status, 'absent') as status
    FROM 
        students s
    JOIN 
        student_subjects ss ON s.id = ss.student_id
    LEFT JOIN 
        attendance a ON s.id = a.student_id AND a.date = ? AND a.subject_id = ?
    WHERE 
        ss.subject_id = ?
    ORDER BY 
        s.roll_no
    ''', (date, subject_id, subject_id))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_class_attendance_report(date):
    """Get attendance report for all subjects on a specific date"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        s.id, 
        s.roll_no, 
        s.name,
        subj.code as subject_code,
        subj.name as subject_name,
        COALESCE(a.status, 'absent') as status,
        a.period
    FROM 
        students s
    JOIN 
        student_subjects ss ON s.id = ss.student_id
    JOIN 
        subjects subj ON ss.subject_id = subj.id
    LEFT JOIN 
        attendance a ON s.id = a.student_id AND a.date = ? AND a.subject_id = subj.id
    ORDER BY 
        s.roll_no, subj.code
    ''', (date,))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_student_attendance_report(student_id):
    """Get attendance report for a specific student across all subjects"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        a.date, 
        s.code as subject_code, 
        s.name as subject_name, 
        a.period, 
        a.status
    FROM 
        attendance a
    JOIN 
        subjects s ON a.subject_id = s.id
    WHERE 
        a.student_id = ?
    ORDER BY 
        a.date DESC, a.period
    ''', (student_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_student_attendance_summary(student_id):
    """Get summary of attendance for a specific student grouped by subject"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        s.id as subject_id,
        s.code as subject_code, 
        s.name as subject_name, 
        COUNT(CASE WHEN a.status = 'present' THEN 1 END) as present_count,
        COUNT(DISTINCT a.date || a.period) as total_classes
    FROM 
        subjects s
    JOIN 
        student_subjects ss ON s.id = ss.subject_id
    LEFT JOIN 
        attendance a ON s.id = a.subject_id AND a.student_id = ss.student_id
    WHERE 
        ss.student_id = ?
    GROUP BY 
        s.id
    ORDER BY 
        s.code
    ''', (student_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_student_details(student_id):
    """Get details for a specific student"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, roll_no, name, email, department, year, division, image_path
    FROM students
    WHERE id = ?
    ''', (student_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "id": result["id"],
            "roll_no": result["roll_no"],
            "name": result["name"],
            "email": result["email"],
            "department": result["department"],
            "year": result["year"],
            "division": result["division"],
            "image_path": result["image_path"]
        }
    else:
        return None

def get_class_attendance_summary(date_from, date_to):
    """Get summary of attendance for all students within a date range"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        s.id as student_id,
        s.roll_no,
        s.name,
        s.division,
        COUNT(CASE WHEN a.status = 'present' THEN 1 END) as present_count,
        COUNT(DISTINCT a.date || a.period || a.subject_id) as total_classes
    FROM 
        students s
    LEFT JOIN 
        attendance a ON s.id = a.student_id AND a.date >= ? AND a.date <= ?
    GROUP BY 
        s.id
    ORDER BY 
        s.roll_no
    ''', (date_from, date_to))
    
    results = cursor.fetchall()
    conn.close()
    
    return results 