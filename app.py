import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import calendar

from utils.db_utils import get_students_by_subject

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="ENTC TY B Facial Attendance System",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    
    /* Metrics styling */
    div.css-12w0qpk.e1tzin5v2 {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1E3A8A;
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1rem;
    }
    
    /* Success metrics */
    .success-metric {
        color: #1cc88a;
        font-weight: bold;
    }
    
    /* Warning metrics */
    .warning-metric {
        color: #f6c23e;
        font-weight: bold;
    }
    
    /* Danger metrics */
    .danger-metric {
        color: #e74a3b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

import os
import datetime
import traceback
from PIL import Image
import pandas as pd
import sqlite3
import requests
import logging
import io
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Create necessary directories if they don't exist
os.makedirs("faces", exist_ok=True)
os.makedirs("excel_exports", exist_ok=True)
os.makedirs("db", exist_ok=True)

# Import database utilities with error handling
try:
    from utils.db_utils import (
        get_connection,
        init_db,
        register_student,
        get_all_students,
        get_subjects,
        mark_attendance,
        get_attendance_report,
        get_class_attendance_report,
        get_student_attendance_report,
        get_student_attendance_summary,
        get_student_details,
        get_class_attendance_summary
    )
except ImportError as e:
    logger.error(f"Error importing database utilities: {str(e)}")
    st.error(f"Database utilities not found: {str(e)}")

# Try importing DeepFace with error handling
try:
    from utils.deepface_utils import (
        verify_faces,
        detect_faces_with_details,
        save_session_stats
    )
    deepface_available = True
except ImportError as e:
    logger.error(f"Error importing DeepFace: {str(e)}")
    deepface_available = False
    st.error(f"Error loading DeepFace module. Please check installation: {str(e)}")

# Initialize the database
try:
    init_db()
except Exception as e:
    logger.error(f"Database initialization error: {str(e)}")
    st.error("Failed to initialize database. Check logs for details.")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Teacher Dashboard"

# Sidebar navigation with buttons
st.sidebar.title("Navigation")

# Modern navigation with icons and better styling
st.sidebar.markdown("### Main")
if st.sidebar.button("📊 Teacher Dashboard", key="nav_dashboard", use_container_width=True):
    st.session_state.page = "Teacher Dashboard"
    
if st.sidebar.button("👤 Student Registration", key="nav_registration", use_container_width=True):
    st.session_state.page = "Student Registration"

if st.sidebar.button("📝 Take Attendance", key="nav_take_attendance", use_container_width=True):
    st.session_state.page = "Take Attendance"

st.sidebar.markdown("### Reports")
if st.sidebar.button("📊 Attendance Reports", key="nav_reports", use_container_width=True):
    st.session_state.page = "Attendance Reports"

if st.sidebar.button("🏫 Class-wise Reports", key="nav_class_reports", use_container_width=True):
    st.session_state.page = "Class Reports"
    
if st.sidebar.button("👨‍🎓 Student-wise Reports", key="nav_student_reports", use_container_width=True):
    st.session_state.page = "Student Reports"

if st.sidebar.button("🔄 Edit Attendance", key="nav_edit_attendance", use_container_width=True):
    st.session_state.page = "Edit Attendance"

st.sidebar.markdown("### Analytics")
if st.sidebar.button("👁️ Recognition Stats", key="nav_stats", use_container_width=True):
    st.session_state.page = "Recognition Stats"

# Get current page from session state
page = st.session_state.page

# Department, Year, Division are fixed as per requirements
department = "ENTC"
year = "TY"
division = "B"

# Subject code to short name mapping
subject_short_names = {
    "Antenna and Wave Propagation": "AWP",
    "Digital Communication": "DC",
    "Computer Networks": "CN",
    "Microprocessors & Microcontrollers": "M&M",
    "Embedded System Design": "ESD",
    "M&M Lab": "MML",
    "DC Lab": "DCL",
    "Mini Project": "MP",
    "Seminar": "SEM"
}

# Helper function to check if a file is an uploaded file
def is_uploaded_file(file_obj):
    return hasattr(file_obj, 'name') and hasattr(file_obj, 'getvalue')

# Function to display recognition statistics
def display_recognition_stats(processing_time, detected_faces, recognized_students, confidence_scores, model_name="Unknown"):
    """Display recognition statistics in a visually appealing way"""
    st.subheader("📊 Recognition Statistics")
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("Processing Time", f"{processing_time:.2f} sec")
    with cols[1]:
        st.metric("Faces Detected", f"{detected_faces}")
    with cols[2]:
        st.metric("Students Recognized", f"{len(recognized_students)}")
    with cols[3]:
        recognition_rate = (len(recognized_students) / detected_faces * 100) if detected_faces > 0 else 0
        st.metric("Recognition Rate", f"{recognition_rate:.1f}%")
    with cols[4]:
        st.metric("Model", model_name)
    
    # Display confidence information if available
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
        
        # Create a histogram of confidence scores
        if len(confidence_scores) > 1:
            hist_data = pd.DataFrame({'Confidence': confidence_scores})
            st.bar_chart(hist_data.Confidence.value_counts(bins=5, sort=False))

# Function to visualize face detection and recognition
def visualize_detected_faces(image_path, face_locations, recognized_students=None):
    """
    Create a visualization of the classroom image with bounding boxes 
    around detected faces, highlighting recognized students
    
    Args:
        image_path: Path to the classroom image
        face_locations: List of face location dictionaries with x, y, w, h
        recognized_students: List of recognized student dictionaries
    """
    try:
        import cv2
        
        # Load image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Failed to load image for visualization: {image_path}")
            return
            
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a copy for drawing
        display_image = image_rgb.copy()
        
        # Draw bounding boxes for all detected faces
        for i, face_loc in enumerate(face_locations):
            x = face_loc.get('x', 0)
            y = face_loc.get('y', 0)
            w = face_loc.get('w', 0)
            h = face_loc.get('h', 0)
            
            # Default color for unrecognized faces (red)
            color = (255, 0, 0)
            label = f"Face {i+1}"
            
            # Check if this face belongs to a recognized student
            if recognized_students:
                # This is a simplified matching approach
                # In a real implementation, you would map detected faces to recognized students
                if i < len(recognized_students):
                    # For recognized students, use green color
                    color = (0, 255, 0)
                    student = recognized_students[i]
                    label = f"{student['roll_no']}"
            
            # Draw rectangle
            cv2.rectangle(display_image, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            cv2.putText(
                display_image, 
                label, 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        # Display the image with annotations
        st.subheader("Visualization of Detected Faces")
        st.image(display_image, caption="Detected Faces", use_column_width=True)
        
    except Exception as e:
        logger.error(f"Error visualizing face detection: {str(e)}")
        st.error(f"Failed to create visualization: {str(e)}")

# Main page content
if page == "Teacher Dashboard":
    st.title("📊 Teacher Dashboard")
    st.markdown('<div class="dashboard-card"><p>View and manage student information for ENTC TY B division.</p></div>', unsafe_allow_html=True)
    st.subheader(f"Department: {department} | Year: {year} | Division: {division}")
    
    # Get all students
    try:
        all_students = get_all_students()
        
        if not all_students:
            st.warning("No students registered yet. Use the Student Registration page to add students.")
        else:
            # Convert to DataFrame for display
            students_df = pd.DataFrame([
                {
                    "ID": student["id"],
                    "Roll No": student["roll_no"],
                    "Name": student["name"],
                    "Email": student["email"] or "",
                    "Department": student["department"],
                    "Year": student["year"],
                    "Division": student["division"]
                }
                for student in all_students
            ])
            
            # Search functionality
            search_term = st.text_input("🔍 Search students by name or roll number:")
            
            if search_term:
                filtered_df = students_df[
                    students_df["Name"].str.contains(search_term, case=False) | 
                    students_df["Roll No"].str.contains(search_term, case=False)
                ]
            else:
                filtered_df = students_df
            
            # Display student statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", len(students_df))
            with col2:
                st.metric("Departments", len(students_df["Department"].unique()))
            with col3:
                st.metric("Divisions", len(students_df["Division"].unique()))
            
            # Display students table
            st.markdown("### Student List")
            st.dataframe(
                filtered_df.drop(columns=["ID"]), 
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            if st.button("Export Student List to Excel"):
                # Create Excel file
                excel_path = os.path.join("excel_exports", f"Students_{department}_{year}{division}.xlsx")
                filtered_df.drop(columns=["ID"]).to_excel(excel_path, index=False)
                
                with open(excel_path, "rb") as file:
                    st.download_button(
                        label="Download Excel File",
                        data=file,
                        file_name=f"Students_{department}_{year}{division}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Student details and editing option
            st.markdown("### Student Details")
            
            if not filtered_df.empty:
                selected_roll = st.selectbox(
                    "Select a student to view details:",
                    options=filtered_df["Roll No"].tolist(),
                    format_func=lambda x: f"{x} - {filtered_df[filtered_df['Roll No']==x]['Name'].values[0]}" if x else "Select a student"
                )
                
                if selected_roll:
                    try:
                        student_id = filtered_df[filtered_df["Roll No"] == selected_roll]["ID"].values[0]
                        selected_student = next((s for s in all_students if s["id"] == student_id), None)
                        
                        if selected_student:
                            cols = st.columns([1, 3])
                            
                            with cols[0]:
                                try:
                                    if os.path.exists(selected_student["image_path"]):
                                        img = Image.open(selected_student["image_path"])
                                        st.image(img, width=200, caption=f"{selected_roll}")
                                    else:
                                        st.warning("Student image file not found.")
                                except Exception as e:
                                    logger.error(f"Error loading student image: {str(e)}")
                                    st.warning("Unable to load student image.")
                            
                            with cols[1]:
                                st.markdown(f"### {selected_student['name']}")
                                st.markdown(f"**Roll No:** {selected_student['roll_no']}")
                                st.markdown(f"**Email:** {selected_student['email'] or 'N/A'}")
                                st.markdown(f"**Department:** {selected_student['department']}")
                                st.markdown(f"**Year:** {selected_student['year']}")
                                st.markdown(f"**Division:** {selected_student['division']}")
                                
                                # Get student's enrolled subjects
                                try:
                                    conn = get_connection()
                                    cursor = conn.cursor()
                                    
                                    cursor.execute('''
                                        SELECT s.code, s.name
                                        FROM subjects s
                                        JOIN student_subjects ss ON s.id = ss.subject_id
                                        WHERE ss.student_id = ?
                                    ''', (student_id,))
                                    
                                    enrolled_subjects = cursor.fetchall()
                                    conn.close()
                                    
                                    if enrolled_subjects:
                                        st.markdown("**Enrolled Subjects:**")
                                        
                                        # Display subjects as badges in a more modern way
                                        subject_badges = []
                                        for subject in enrolled_subjects:
                                            subject_code = subject[0]
                                            subject_name = subject[1]
                                            short_name = subject_short_names.get(subject_name, subject_code)
                                            subject_badges.append(f"<span style='display:inline-block; margin:2px; padding:4px 8px; background-color:#1E3A8A; color:white; border-radius:12px; font-size:0.8em;'>{short_name}</span>")
                                        
                                        st.markdown(f"<div>{''.join(subject_badges)}</div>", unsafe_allow_html=True)
                                        st.caption(f"Total subjects: {len(enrolled_subjects)}")
                                        
                                        # Add a section to show subject-wise attendance if available
                                        try:
                                            attendance_summary = get_student_attendance_summary(student_id)
                                            if attendance_summary:
                                                st.markdown("**Attendance Overview:**")
                                                
                                                # Calculate overall attendance
                                                total_present = sum(row["present_count"] for row in attendance_summary)
                                                total_classes = sum(row["total_classes"] for row in attendance_summary)
                                                overall_percentage = round(total_present / total_classes * 100, 1) if total_classes > 0 else 0
                                                
                                                # Display overall attendance with appropriate color
                                                color = "#4caf50" if overall_percentage >= 75 else "#ff9800" if overall_percentage >= 50 else "#f44336"
                                                st.markdown(f"<div style='font-weight:bold'>Overall Attendance: <span style='color:{color}'>{overall_percentage}%</span></div>", unsafe_allow_html=True)
                                                
                                        except Exception as att_e:
                                            logger.error(f"Error retrieving attendance summary: {str(att_e)}")
                                            # Just skip attendance display if there's an error
                                            pass
                                    else:
                                        st.warning("Student is not enrolled in any subjects.")
                                except Exception as e:
                                    logger.error(f"Error retrieving enrolled subjects: {str(e)}")
                                    st.error("Failed to retrieve enrolled subjects.")
                        else:
                            st.error("Student details not found in database.")
                    except Exception as e:
                        logger.error(f"Error retrieving student details: {str(e)}")
                        st.error(f"Failed to retrieve student details. Error: {str(e)}")
            else:
                st.info("No students match your search criteria.")
    
    except Exception as e:
        logger.error(f"Error loading student data: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error: {str(e)}")

elif page == "Student Registration":
    st.title("👤 Student Registration")
    
    with st.form("student_registration"):
        name = st.text_input("Full Name")
        roll_no = st.text_input("Roll Number (e.g., EC3201)")
        email = st.text_input("Email Address")
        
        # Fixed fields
        st.text(f"Department: {department}")
        st.text(f"Year: {year}")
        st.text(f"Division: {division}")
        
        # Subject selection
        try:
            subjects = get_subjects()
            subject_options = [subject[1] for subject in subjects]  # Use code instead of full name
            
            # Add select all option
            st.subheader("Subject Enrollment")
            select_all = st.checkbox("Select All Subjects", value=False)
            
            if select_all:
                selected_subjects = subject_options
                st.info(f"All {len(subject_options)} subjects selected")
                # Show the list of selected subjects for clarity
                st.write("Enrolled in:", ", ".join(selected_subjects))
            else:
                selected_subjects = st.multiselect("Select Enrolled Subjects", subject_options)
                
        except Exception as e:
            logger.error(f"Error loading subjects: {str(e)}")
            st.error("Failed to load subjects. Please check database connection.")
            subjects = []
            selected_subjects = []
            select_all = False
        
        # Face image upload
        face_image = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
        
        # Preview image if uploaded
        if face_image is not None:
            try:
                st.image(face_image, caption="Student Face", width=200)
            except Exception as e:
                logger.error(f"Error displaying face image: {str(e)}")
                st.error("Failed to display image preview.")
        
        submitted = st.form_submit_button("Register Student")
        
        if submitted:
            if not name or not roll_no or not face_image or not selected_subjects:
                st.error("All fields are required except email")
            else:
                try:
                    # Save face image
                    image_path = os.path.join("faces", f"{roll_no}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(face_image.getvalue())
                    
                    # Get subject IDs from selected subject names
                    subject_ids = [subject[0] for subject in subjects if subject[1] in selected_subjects]
                    
                    # Register student in database
                    student_id = register_student(roll_no, name, department, year, division, image_path, subject_ids, email)
                    
                    if student_id:
                        st.success(f"Student {name} registered successfully!")
                        if select_all:
                            st.success(f"Enrolled in all {len(subject_options)} subjects")
                        else:
                            st.success(f"Enrolled in {len(selected_subjects)} subjects")
                    else:
                        st.error("Failed to register student. Roll number may already exist.")
                except Exception as e:
                    logger.error(f"Error registering student: {str(e)}")
                    st.error(f"Error registering student: {str(e)}")

elif page == "Take Attendance":
    st.title("📝 Take Attendance")
    st.markdown('<div class="dashboard-card"><p>Capture attendance using facial recognition.</p></div>', unsafe_allow_html=True)
    st.subheader(f"Department: {department} | Year: {year} | Division: {division}")
    
    # Check if we need to show success message from previous submission
    if hasattr(st.session_state, 'last_attendance') and st.session_state.get('clear_attendance_form', False):
        last = st.session_state.last_attendance
        st.success(f"✅ Successfully saved attendance for {last['count']} students in {last['subject']} on {last['date']} ({last['period']}).")
        
        # Clear the flag so the message doesn't show again on manual refresh
        st.session_state.clear_attendance_form = False
        
        # Reset form fields
        if 'form_key' in st.session_state:
            del st.session_state['form_key']
        
        # Clear uploaded files
        if 'uploaded_files' in st.session_state:
            del st.session_state['uploaded_files']
        
        # Add a button to take new attendance
        if st.button("Take Another Attendance", key="take_another"):
            # This will be handled in the next rerun - just clear the session state
            pass
    
    # Create columns for form inputs
    col1, col2 = st.columns(2)
    
    # Get subjects from database with error handling
    try:
        subjects = get_subjects()
        subject_options = [subject[1] for subject in subjects]  # Use code instead of full name
        
        with col1:
            if subject_options:
                selected_subject = st.selectbox("Select Subject", subject_options)
                subject_id = next((subject[0] for subject in subjects if subject[1] == selected_subject), None)
            else:
                st.warning("No subjects found in database. Please check database configuration.")
                selected_subject = None
                subject_id = None
        
        with col2:
            periods = [
                "10:15 - 11:15",
                "11:15 - 12:15",
                "01:15 - 02:15",
                "02:15 - 03:15",
                "03:30 - 04:30",
                "04:30 - 05:30"
            ]
            selected_period = st.selectbox("Select Period", periods)
        
        # Add date selection
        attendance_date = st.date_input("Select Date", value=datetime.date.today())
        
    except Exception as e:
        logger.error(f"Error retrieving subjects: {str(e)}")
        st.error("Failed to retrieve subjects from database.")
        subjects = []
        subject_options = []
        attendance_date = datetime.date.today()
    
    st.divider()
    
    # Image input section
    st.subheader("Classroom Image")
    image_source = st.radio(
        "Image Source",
        ["Upload Image", "Capture from ESP32-CAM"],
        horizontal=True
    )
    
    # Add model selection options
    st.subheader("Recognition Settings")
    cols = st.columns([2, 1])
    with cols[0]:
        model_name = st.selectbox(
            "Face Recognition Model",
            ["Facenet512", "VGG-Face", "OpenFace", "DeepFace", "ArcFace", "SFace"],
            index=0,
            help="Facenet512 provides higher accuracy but may be slightly slower"
        )

    with cols[1]:
        threshold = st.slider(
            "Matching Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Lower values mean stricter matching (fewer false positives)"
        )
    
    image_files = []
    if image_source == "Upload Image":
        allow_multiple = st.checkbox("Upload multiple images", value=False, 
                                    help="Enable to upload multiple classroom images at once")
        
        if allow_multiple:
            # Multiple image upload
            uploaded_files = st.file_uploader("Upload classroom images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi_image_uploader")
            
            if uploaded_files:
                image_files = uploaded_files
                # Display image previews
                st.subheader("Uploaded Images")
                preview_cols = st.columns(min(len(uploaded_files), 3))
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with preview_cols[i % 3]:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=f"Image {i+1}", width=200)
                    except Exception as e:
                        st.error(f"Error opening image {i+1}: {str(e)}")
                
                # Select which image to process
                if len(uploaded_files) > 1:
                    selected_image_index = st.selectbox(
                        "Select which image to process", 
                        range(len(uploaded_files)), 
                        format_func=lambda i: f"Image {i+1}"
                    )
                    image_file = uploaded_files[selected_image_index]
                else:
                    image_file = uploaded_files[0]
                    
                # Add button to process all images
                process_all = st.checkbox("Process all images", value=False,
                                         help="Enable to analyze all uploaded images in sequence")
        else:
            # Single image upload (original behavior)
            image_file = st.file_uploader("Upload classroom image", type=["jpg", "jpeg", "png"], key="single_image_uploader")
            image_files = [image_file] if image_file is not None else []
            process_all = False
            
            if image_file is not None:
                try:
                    # Display uploaded image
                    image = Image.open(image_file)
                    st.image(image, caption="Uploaded Image", width=400)
                except Exception as e:
                    logger.error(f"Error opening uploaded image: {str(e)}")
                    st.error("Failed to open uploaded image. Please try a different file.")
    else:
        esp32_url = st.text_input("ESP32-CAM URL", value="http://esp32-cam-ip/capture")
        if st.button("Capture Image"):
            try:
                with st.spinner("Capturing image from ESP32-CAM..."):
                    response = requests.get(esp32_url, timeout=10)
                    if response.status_code == 200:
                        # Save the captured image
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f"captured_{timestamp}.jpg"
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        
                        image = Image.open(image_path)
                        st.image(image, caption="Captured Image", width=400)
                        image_file = image_path
                        image_files = [image_path]  # Add to image_files list for consistency
                    else:
                        st.error(f"Failed to capture image from ESP32-CAM. Status code: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"ESP32-CAM connection error: {str(e)}")
                st.error(f"Error connecting to ESP32-CAM: {str(e)}")
            except Exception as e:
                logger.error(f"Error capturing image: {str(e)}")
                st.error(f"Error: {str(e)}")
    
    # Process attendance button - just analyze, don't save yet
    if deepface_available and st.button("Analyze Image") and len(image_files) > 0 and subject_id is not None:
        # If processing all images is enabled and there are multiple images
        if process_all and len(image_files) > 1:
            st.subheader("Processing Multiple Images")
            
            # Create a container for all results
            all_results_container = st.container()
            
            # Initialize combined results
            all_detected_faces = 0
            all_recognized_students = set()  # Use a set to avoid duplicates
            all_confidence_scores = []
            total_processing_time = 0
            
            # Process each image
            for idx, img_file in enumerate(image_files):
                with st.spinner(f"Processing image {idx+1} of {len(image_files)}..."):
                    try:
                        # Record start time for performance metrics
                        start_time = time.time()
                        
                        # Prepare parameters for DeepFace verification
                        temp_image_path = None
                        
                        if is_uploaded_file(img_file):
                            # For uploaded files, save to a temporary file
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            temp_image_path = f"temp_image_{idx}_{timestamp}.jpg"
                            with open(temp_image_path, "wb") as f:
                                f.write(img_file.getvalue())
                            classroom_image = temp_image_path
                        else:
                            # For paths from ESP32-CAM
                            classroom_image = img_file
                        
                        # Detect all faces first to get count
                        detected_faces, face_locations = detect_faces_with_details(classroom_image)
                        
                        # Use the selected model with adjusted threshold for better recognition
                        present_students, confidence_scores = verify_faces(
                            classroom_image_path=classroom_image, 
                            students=get_all_students(),
                            threshold=threshold,
                            model_name=model_name,
                            return_confidence=True
                        )
                        
                        # Calculate processing time
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # Update combined results
                        all_detected_faces += len(detected_faces)
                        all_recognized_students.update([student["id"] for student in present_students])
                        all_confidence_scores.extend(confidence_scores)
                        total_processing_time += processing_time
                        
                        # Display individual image results
                        st.write(f"### Image {idx+1} Results")
                        st.write(f"Detected faces: {len(detected_faces)}")
                        st.write(f"Recognized students: {len(present_students)}")
                        st.write(f"Processing time: {processing_time:.2f} seconds")
                        
                        # Clean up temporary file if created
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            
                    except Exception as e:
                        logger.error(f"Error processing image {idx+1}: {str(e)}\n{traceback.format_exc()}")
                        st.error(f"Error processing image {idx+1}: {str(e)}")
            
            # Display combined results
            with all_results_container:
                st.subheader("Combined Results")
                
                # Display metrics
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Total Processing Time", f"{total_processing_time:.2f} sec")
                with cols[1]:
                    st.metric("Total Faces Detected", f"{all_detected_faces}")
                with cols[2]:
                    st.metric("Total Students Recognized", f"{len(all_recognized_students)}")
                with cols[3]:
                    recognition_rate = (len(all_recognized_students) / all_detected_faces * 100) if all_detected_faces > 0 else 0
                    st.metric("Overall Recognition Rate", f"{recognition_rate:.1f}%")
                
                # Display recognized students
                st.markdown("### All Recognized Students")
                
                # Get full details of all recognized students
                recognized_student_details = []
                for student_id in all_recognized_students:
                    student_details = get_student_details(student_id)
                    if student_details:
                        recognized_student_details.append(student_details)
                
                # Display in a grid
                if recognized_student_details:
                    grid_cols = st.columns(4)
                    for i, student in enumerate(recognized_student_details):
                        with grid_cols[i % 4]:
                            try:
                                img = Image.open(student["image_path"])
                                st.image(img, caption=f"{student['roll_no']}\n{student['name']}", width=150)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                    
                    # Allow marking attendance for all recognized students
                    st.markdown("### Mark Attendance")
                    st.markdown("Select students to mark as present:")
                    
                    with st.form(key=f"batch_attendance_form_{int(time.time())}"):
                        selected_students = {}
                        
                        for student in recognized_student_details:
                            selected_students[student["id"]] = st.checkbox(
                                f"{student['roll_no']} - {student['name']}",
                                value=True,
                                key=f"batch_student_{student['id']}"
                            )
                        
                        # Save attendance button
                        submit_attendance = st.form_submit_button("Save Attendance")
                        
                        if submit_attendance:
                            # Get selected students
                            marked_ids = [student_id for student_id, selected in selected_students.items() if selected]
                            
                            if marked_ids:
                                # Mark attendance in database
                                selected_date = attendance_date.strftime("%Y-%m-%d")
                                success_count = 0
                                
                                for student_id in marked_ids:
                                    success = mark_attendance(student_id, subject_id, selected_date, selected_period)
                                    if success:
                                        success_count += 1
                                
                                if success_count > 0:
                                    st.success(f"Attendance saved for {success_count} students on {selected_date}!")
                                    
                                    # Store attendance data in session state before clearing
                                    st.session_state.last_attendance = {
                                        'subject': selected_subject,
                                        'date': selected_date,
                                        'period': selected_period,
                                        'count': success_count
                                    }
                                    
                                    # Clear form by triggering rerun with parameter
                                    st.session_state.clear_attendance_form = True
                                    
                                    # Generate a new unique form key for next time
                                    st.session_state.form_key = f"batch_attendance_form_{int(time.time())}"
                                    
                                    # Clear uploaded files to reset the form
                                    st.session_state.uploaded_files = None
                                    
                                    # Add button to take new attendance
                                    st.button("Take New Attendance", on_click=lambda: st.session_state.update({
                                        'clear_attendance_form': False,
                                        'page': 'Take Attendance'
                                    }))
                                else:
                                    st.error("Failed to save attendance. Please try again.")
                            else:
                                st.warning("No students were selected for attendance.")
                else:
                    st.warning("No students were recognized in any of the images.")
        else:
            # Process single image (existing code)
            with st.spinner("Processing image using facial recognition..."):
                try:
                    # Record start time for performance metrics
                    start_time = time.time()
                    
                    # Get all students enrolled in the selected subject
                    all_students = get_all_students()
                    
                    # Check if we have students in database
                    if not all_students:
                        st.warning("No students found in database. Please register students first.")
                    else:
                        # Prepare parameters for DeepFace verification
                        # Handle both uploaded file objects and file paths
                        temp_image_path = None
                        
                        if is_uploaded_file(image_file):
                            # For uploaded files, save to a temporary file
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            temp_image_path = f"temp_image_{timestamp}.jpg"
                            with open(temp_image_path, "wb") as f:
                                f.write(image_file.getvalue())
                            classroom_image = temp_image_path
                        else:
                            # For paths from ESP32-CAM
                            classroom_image = image_file
                        
                        # Detect all faces first to get count
                        detected_faces, face_locations = detect_faces_with_details(classroom_image)
                        
                        # Use the selected model with adjusted threshold for better recognition
                        present_students, confidence_scores = verify_faces(
                            classroom_image_path=classroom_image, 
                            students=all_students,
                            threshold=threshold,
                            model_name=model_name,
                            return_confidence=True
                        )
                        
                        # Calculate processing time
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # Save statistics to session state
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                        recognition_rate = (len(present_students) / len(detected_faces) * 100) if detected_faces else 0
                        
                        stats = {
                            'datetime': current_time,
                            'subject': selected_subject,
                            'period': selected_period,
                            'processing_time': processing_time,
                            'detected_faces': len(detected_faces),
                            'recognized_students': len(present_students),
                            'recognition_rate': recognition_rate,
                            'avg_confidence': avg_confidence
                        }
                        
                        save_session_stats(stats)
                        
                        # Display statistics
                        display_recognition_stats(
                            processing_time=processing_time,
                            detected_faces=len(detected_faces),
                            recognized_students=present_students,
                            confidence_scores=confidence_scores,
                            model_name=model_name
                        )
                        
                        # Display face detection visualization
                        visualize_detected_faces(classroom_image, face_locations, present_students)
                        
                        # Display attendance summary
                        st.subheader("Attendance Summary")
                        
                        # Show comparison of detected vs recognized faces
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"**Detected Faces:** {len(detected_faces)}")
                            
                        with col2:
                            st.success(f"**Recognized Students:** {len(present_students)}")
                        
                        if len(detected_faces) > len(present_students):
                            st.warning(f"⚠️ {len(detected_faces) - len(present_students)} faces detected but not recognized. These may be students not registered in the system or false detections.")
                        
                        # Show table of recognized students with a checkbox to include/exclude
                        if present_students:
                            st.markdown("### Recognized Students")
                            st.markdown("You can uncheck any students that were incorrectly recognized before saving attendance.")
                            
                            # Initialize a key for form state
                            form_key = f"attendance_form_{int(time.time())}"
                            
                            with st.form(key=form_key):
                                # Create checkboxes for each student
                                selected_students = {}
                                
                                for i, student in enumerate(present_students):
                                    selected_students[student["id"]] = st.checkbox(
                                        f"{student['roll_no']} - {student['name']}",
                                        value=True,
                                        key=f"student_{student['id']}"
                                    )
                                
                                # Save attendance button
                                submit_attendance = st.form_submit_button("Save Attendance")
                                
                                if submit_attendance:
                                    # Get selected students
                                    marked_ids = [student_id for student_id, selected in selected_students.items() if selected]
                                    
                                    if marked_ids:
                                        # Mark attendance in database
                                        selected_date = attendance_date.strftime("%Y-%m-%d")
                                        success_count = 0
                                        
                                        for student_id in marked_ids:
                                            success = mark_attendance(student_id, subject_id, selected_date, selected_period)
                                            if success:
                                                success_count += 1
                                        
                                        if success_count > 0:
                                            st.success(f"Attendance saved for {success_count} students on {selected_date}!")
                                            
                                            # Store attendance data in session state before clearing
                                            st.session_state.last_attendance = {
                                                'subject': selected_subject,
                                                'date': selected_date,
                                                'period': selected_period,
                                                'count': success_count
                                            }
                                            
                                            # Clear form by triggering rerun with parameter
                                            st.session_state.clear_attendance_form = True
                                            
                                            # Generate a new unique form key for next time
                                            st.session_state.form_key = f"attendance_form_{int(time.time())}"
                                            
                                            # Clear uploaded files to reset the form
                                            st.session_state.uploaded_files = None
                                            
                                            # Add button to take new attendance
                                            st.button("Take New Attendance", on_click=lambda: st.session_state.update({
                                                'clear_attendance_form': False,
                                                'page': 'Take Attendance'
                                            }))
                                        else:
                                            st.error("Failed to save attendance. Please try again.")
                                    else:
                                        st.warning("No students were selected for attendance.")
                            
                            # Visual representation of recognized students
                            st.markdown("### Recognized Students")
                            cols = st.columns(4)
                            col_idx = 0
                            
                            for student in present_students:
                                # Display student face image
                                with cols[col_idx]:
                                    try:
                                        img = Image.open(student["image_path"])
                                        st.image(img, caption=f"{student['roll_no']}\n{student['name']}", width=150)
                                        
                                        # Add confidence score if available
                                        idx = present_students.index(student)
                                        if idx < len(confidence_scores):
                                            st.caption(f"Confidence: {confidence_scores[idx]:.2f}")
                                    except Exception as e:
                                        st.error(f"Error loading image: {str(e)}")
                                    
                                    # Move to next column
                                    col_idx = (col_idx + 1) % 4
                        
                        # Clean up temporary file if created
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                    
                except Exception as e:
                    logger.error(f"Error processing attendance: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Error processing attendance: {str(e)}")
    elif not deepface_available and st.button("Analyze Image"):
        st.error("DeepFace module is not available. Please check installation and dependencies.")

elif page == "Attendance Reports":
    st.title("📊 Attendance Reports")
    
    try:
        # Get subjects from database
        subjects = get_subjects()
        subject_options = [subject[1] for subject in subjects]  # Use code instead of full name
        
        if subject_options:
            selected_subject = st.selectbox("Select Subject", subject_options)
            subject_id = next((subject[0] for subject in subjects if subject[1] == selected_subject), None)
            
            # Date selection
            report_date = st.date_input("Select Date", value=datetime.date.today())
            
            # Get attendance report
            if subject_id:
                attendance_data = get_attendance_report(subject_id, report_date.strftime("%Y-%m-%d"))
                
                if attendance_data:
                    st.subheader(f"Attendance for {selected_subject} on {report_date}")
                    
                    try:
                        # Create DataFrame for display
                        df = pd.DataFrame(
                            [(data[1], data[2], data[3], "✅" if data[4] == "present" else "❌") for data in attendance_data],
                            columns=["Roll No", "Name", "Email", "Status"]
                        )
                        st.dataframe(df)
                        
                        # Calculate and display statistics
                        present_count = df[df['Status'] == '✅'].shape[0]
                        absent_count = df[df['Status'] == '❌'].shape[0]
                        total_count = present_count + absent_count
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Students", str(total_count))
                        with col2:
                            st.metric("Present", str(present_count))
                        with col3:
                            st.metric("Absent", str(absent_count))
                        with col4:
                            attendance_percentage = (present_count / total_count * 100) if total_count > 0 else 0
                            st.metric("Attendance %", f"{attendance_percentage:.1f}%")
                        
                        # Show pie chart
                        chart_data = pd.DataFrame({
                            'Status': ['Present', 'Absent'],
                            'Count': [present_count, absent_count]
                        })
                        if present_count > 0 or absent_count > 0:
                            st.subheader("Attendance Visualization")
                            st.bar_chart(chart_data.set_index('Status'))
                        
                        # Export to Excel
                        excel_path = os.path.join("excel_exports", f"{selected_subject}_{department}_{year}{division}.xlsx")
                        
                        # Create pivot table for Excel export
                        pivot_data = []
                        for data in attendance_data:
                            pivot_data.append({
                                "Roll No": data[1],
                                "Name": data[2],
                                "Email": data[3],
                                report_date.strftime("%Y-%m-%d"): "✅" if data[4] == "present" else "❌"
                            })
                        
                        pivot_df = pd.DataFrame(pivot_data)
                        
                        try:
                            # Check if file exists already
                            if os.path.exists(excel_path):
                                # Read existing Excel and merge with new data
                                try:
                                    existing_df = pd.read_excel(excel_path)
                                    
                                    # Merge the dataframes
                                    merged_df = pd.merge(existing_df, pivot_df, on=["Roll No", "Name", "Email"], how="outer")
                                    merged_df.to_excel(excel_path, index=False)
                                except Exception as excel_read_error:
                                    logger.error(f"Error reading existing Excel: {str(excel_read_error)}")
                                    # If existing file has issues, just write the new one
                                    pivot_df.to_excel(excel_path, index=False)
                            else:
                                pivot_df.to_excel(excel_path, index=False)
                            
                            with open(excel_path, "rb") as file:
                                st.download_button(
                                    label="Download Excel Report",
                                    data=file,
                                    file_name=f"{selected_subject}_{report_date}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        except Exception as excel_error:
                            logger.error(f"Error exporting to Excel: {str(excel_error)}")
                            st.error(f"Error exporting to Excel: {str(excel_error)}")
                        
                        # Option to notify absent students
                        absent_students = df[df["Status"] == "❌"]
                        if not absent_students.empty and st.button("Send Email Notifications to Absent Students"):
                            st.info("Email notification feature is under development.")
                            # In a real implementation, this would integrate with an email service
                            # to send notifications to the absent students
                            for _, student in absent_students.iterrows():
                                if student["Email"]:
                                    st.write(f"Would send notification to: {student['Email']}")
                    except Exception as df_error:
                        logger.error(f"Error processing attendance data: {str(df_error)}")
                        st.error(f"Error processing attendance data: {str(df_error)}")
                else:
                    st.info(f"No attendance records found for {selected_subject} on {report_date}")
        else:
            st.warning("No subjects found in database. Please check database configuration.")
    except Exception as e:
        logger.error(f"Error loading attendance report page: {str(e)}")
        st.error(f"Error: {str(e)}")

elif page == "Recognition Stats":
    st.title("👁️ Facial Recognition Analytics")
    
    st.write("""
    This page provides detailed analytics about the facial recognition system performance.
    View statistics on recognition accuracy, processing times, and system performance.
    """)
    
    # Get latest stats if available
    if 'recognition_stats' in st.session_state:
        stats = st.session_state.recognition_stats
        
        st.subheader("Overall System Performance")
        
        # Display metrics in a nice layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg. Processing Time", f"{stats.get('avg_processing_time', 0):.2f} sec")
        with col2:
            st.metric("Recognition Rate", f"{stats.get('recognition_rate', 0):.1f}%")
        with col3:
            st.metric("Avg. Confidence", f"{stats.get('avg_confidence', 0):.2f}")
        with col4:
            st.metric("Total Sessions", str(stats.get('total_sessions', 0)))
        
        # Display historical data if available
        if 'history' in stats and len(stats['history']) > 0:
            st.subheader("Historical Performance")
            
            # Create DataFrame from history
            history_df = pd.DataFrame(stats['history'])
            
            # Display line charts
            if 'recognition_rate' in history_df.columns:
                st.line_chart(history_df[['recognition_rate']])
            if 'processing_time' in history_df.columns:
                st.line_chart(history_df[['processing_time']])
            
        # Display most recent recognition details
        if 'last_session' in stats:
            last = stats['last_session']
            st.subheader("Last Recognition Session")
            
            st.write(f"Date/Time: {last.get('datetime', 'Unknown')}")
            st.write(f"Subject: {last.get('subject', 'Unknown')}")
            st.write(f"Period: {last.get('period', 'Unknown')}")
            
            # Display recognized vs. not recognized
            recognized = last.get('recognized_count', 0)
            not_recognized = last.get('total_faces', 0) - recognized
            
            chart_data = pd.DataFrame({
                'Status': ['Recognized', 'Not Recognized'],
                'Count': [recognized, not_recognized]
            })
            if recognized > 0 or not_recognized > 0:
                st.bar_chart(chart_data.set_index('Status'))
        
        # Add detailed accuracy metrics section
        st.subheader("Detailed Accuracy Metrics")
        
        tabs = st.tabs(["Confusion Matrix", "Precision & Recall", "Accuracy Over Time"])
        
        with tabs[0]:
            # Confusion Matrix visualization
            st.write("Confusion Matrix")
            if 'confusion_matrix' in stats:
                cm = stats['confusion_matrix']
                
                try:
                    # Create a basic confusion matrix visualization
                    cm_data = np.array([[cm.get('TP', 0), cm.get('FP', 0)], 
                                      [cm.get('FN', 0), cm.get('TN', 0)]])
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Present', 'Absent'],
                              yticklabels=['Present', 'Absent'])
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig)
                except Exception as e:
                    logger.error(f"Error creating confusion matrix visualization: {str(e)}")
                    st.error("Could not create confusion matrix visualization. Make sure matplotlib and seaborn are installed.")
                    
                    # Fallback to text display
                    st.write("Confusion Matrix Data:")
                    st.write(f"True Positives: {cm.get('TP', 0)}")
                    st.write(f"False Positives: {cm.get('FP', 0)}")
                    st.write(f"False Negatives: {cm.get('FN', 0)}")
                    st.write(f"True Negatives: {cm.get('TN', 0)}")
            else:
                # Create placeholder matrix with sample data
                tp = stats.get('total_students_recognized', 0)
                fn = stats.get('total_faces_detected', 0) - tp
                
                cm_data = pd.DataFrame({
                    '': ['Actual Present', 'Actual Absent'],
                    'Predicted Present': [tp, 0],  # We don't track false positives yet
                    'Predicted Absent': [fn, 0]    # We don't track true negatives yet
                })
                
                st.write("Detected Faces vs. Recognized Students:")
                st.dataframe(cm_data.set_index(''))
                st.info("Note: Complete confusion matrix tracking requires manual verification of results.")
        
        with tabs[1]:
            # Calculate precision and recall if we have enough data
            if stats.get('total_students_recognized', 0) > 0:
                # Simplified calculation based on available data
                precision = 1.0  # Assuming all recognitions are correct
                recall = (stats.get('total_students_recognized', 0) / 
                         stats.get('total_faces_detected', 0) if stats.get('total_faces_detected', 0) > 0 else 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precision", f"{precision:.2f}")
                    st.caption("Correctly identified / Total identified")
                with col2:
                    st.metric("Recall", f"{recall:.2f}")
                    st.caption("Correctly identified / Total actual faces")
                
                # F1 Score
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    st.metric("F1 Score", f"{f1:.2f}")
            else:
                st.info("Not enough data to calculate precision and recall metrics.")
        
        with tabs[2]:
            # Display accuracy over time if we have history
            if 'history' in stats and len(stats['history']) > 0:
                history_df = pd.DataFrame(stats['history'])
                if 'datetime' in history_df and 'recognition_rate' in history_df:
                    try:
                        history_df['datetime'] = pd.to_datetime(history_df['datetime'])
                        
                        # Plot accuracy over time
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(history_df['datetime'], history_df['recognition_rate'], 'o-', linewidth=2)
                        ax.set_xlabel('Date/Time')
                        ax.set_ylabel('Recognition Rate (%)')
                        ax.set_title('Face Recognition Accuracy Over Time')
                        ax.grid(True)
                        
                        if len(history_df) > 5:
                            plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                    except Exception as e:
                        logger.error(f"Error creating accuracy over time plot: {str(e)}")
                        st.error("Could not create accuracy plot. Using simplified view instead.")
                        
                        # Fallback to simpler visualization
                        st.write("Recognition Rate Over Time:")
                        if 'datetime' in history_df.columns and 'recognition_rate' in history_df.columns:
                            simple_df = history_df[['datetime', 'recognition_rate']].copy()
                            simple_df['datetime'] = simple_df['datetime'].astype(str)
                            simple_df = simple_df.set_index('datetime')
                            st.line_chart(simple_df)
                else:
                    st.info("Not enough historical data with timestamps to plot accuracy over time.")
            else:
                st.info("No historical data available to show accuracy trends.")
    else:
        st.info("No recognition statistics available yet. Process attendance first to generate statistics.")
        
        # Show sample data to demonstrate the feature
        st.subheader("Sample Statistics (Demo)")
        
        # Create sample metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg. Processing Time", "1.25 sec")
        with col2:
            st.metric("Recognition Rate", "85.0%")
        with col3:
            st.metric("Avg. Confidence", "0.78")
        with col4:
            st.metric("Total Sessions", "0")
            
        st.caption("This is a demonstration of how the statistics will appear after processing attendance.")
        
        # Add instructions
        st.markdown("""
        ### How to generate recognition statistics:
        1. Go to the **Take Attendance** page
        2. Upload a classroom image or capture from camera
        3. Click **Analyze Image** to process the attendance
        4. The statistics will be automatically saved and displayed here
        """)
        
        # Add sample chart
        st.subheader("Sample Recognition Rate Chart (Demo)")
        sample_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'rate': [75, 82, 78, 88, 85]
        }).set_index('date')
        st.line_chart(sample_data)

elif page == "Class Reports":
    st.title("🏫 Class-wise Attendance Reports")
    st.markdown('<div class="dashboard-card"><p>View attendance statistics for the entire class across all subjects.</p></div>', unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range selection
            date_from = st.date_input("From Date", value=datetime.date.today() - datetime.timedelta(days=14))
        
        with col2:
            date_to = st.date_input("To Date", value=datetime.date.today())
            
        if date_from > date_to:
            st.error("Error: End date must fall after start date.")
        else:
            # Get summary statistics
            summary_data = get_class_attendance_summary(
                date_from.strftime("%Y-%m-%d"), 
                date_to.strftime("%Y-%m-%d")
            )
            
            if summary_data:
                # Convert to DataFrame
                df_summary = pd.DataFrame([
                    {
                        "Roll No": row["roll_no"],
                        "Name": row["name"],
                        "Division": row["division"],
                        "Present": row["present_count"],
                        "Total Classes": row["total_classes"],
                        "Attendance %": round(row["present_count"] / row["total_classes"] * 100, 1) if row["total_classes"] > 0 else 0
                    }
                    for row in summary_data
                ])
                
                # Summary metrics
                total_students = len(df_summary)
                avg_attendance = df_summary["Attendance %"].mean() if not df_summary.empty else 0
                below_75_count = len(df_summary[df_summary["Attendance %"] < 75]) if not df_summary.empty else 0
                
                # Display metrics
                st.markdown("### Overall Attendance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Students", str(total_students))
                with col2:
                    st.metric("Avg. Attendance", f"{avg_attendance:.1f}%")
                with col3:
                    st.metric("Below 75%", str(below_75_count))
                with col4:
                    st.metric("Date Range", f"{(date_to - date_from).days + 1} days")
                
                # Attendance distribution
                st.markdown("### Attendance Distribution")
                
                # Create bins for attendance percentage
                bins = [0, 25, 50, 75, 90, 100]
                labels = ["0-25%", "26-50%", "51-75%", "76-90%", "91-100%"]
                
                # Add a new column with binned values
                df_summary["Attendance Range"] = pd.cut(df_summary["Attendance %"], bins=bins, labels=labels, right=True)
                
                # Count students in each bin
                attendance_dist = df_summary["Attendance Range"].value_counts().sort_index()
                
                # Create a color scale based on attendance
                colors = ["#f44336", "#ff9800", "#ffeb3b", "#8bc34a", "#4caf50"]
                
                # Create a bar chart with plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=attendance_dist.index,
                        y=attendance_dist.values,
                        text=attendance_dist.values,
                        textposition='auto',
                        marker_color=colors[:len(attendance_dist)]
                    )
                ])
                
                fig.update_layout(
                    title="Number of Students by Attendance Range",
                    xaxis_title="Attendance Range",
                    yaxis_title="Number of Students",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the full dataset with conditional formatting
                st.markdown("### Student-wise Attendance")
                
                # Add color formatting based on attendance percentage
                def color_attendance(val):
                    if val < 50:
                        return 'color: #f44336; font-weight: bold'  # Red
                    elif val < 75:
                        return 'color: #ff9800; font-weight: bold'  # Orange
                    else:
                        return 'color: #4caf50; font-weight: bold'  # Green
                
                # Apply the formatting to the attendance percentage column
                styled_df = df_summary.style.applymap(
                    color_attendance, 
                    subset=pd.IndexSlice[:, ['Attendance %']]
                )
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Add subject-wise attendance breakdown
                st.markdown("### Subject-wise Attendance")
                
                # Create tabs for each subject
                try:
                    subjects = get_subjects()
                    if subjects:
                        subject_tabs = st.tabs([subject[1] for subject in subjects])
                        
                        for i, subject in enumerate(subjects):
                            with subject_tabs[i]:
                                subject_id = subject[0]
                                subject_name = subject[1]
                                
                                # Get all dates in range
                                dates = []
                                current_date = date_from
                                while current_date <= date_to:
                                    dates.append(current_date.strftime("%Y-%m-%d"))
                                    current_date += datetime.timedelta(days=1)
                                
                                # Create a dictionary to store all attendance records
                                attendance_by_date = {}
                                
                                # For each date, get attendance report
                                for date_str in dates:
                                    attendance_data = get_attendance_report(subject_id, date_str)
                                    if attendance_data:
                                        attendance_by_date[date_str] = {
                                            data[1]: data[4] for data in attendance_data
                                        }
                                
                                if attendance_by_date:
                                    # Create a pivot table with students as rows and dates as columns
                                    rows = []
                                    for student in df_summary.to_dict('records'):
                                        roll_no = student["Roll No"]
                                        name = student["Name"]
                                        
                                        row = {"Roll No": roll_no, "Name": name}
                                        
                                        # Add attendance status for each date
                                        present_count = 0
                                        total_count = 0
                                        
                                        for date_str in dates:
                                            if date_str in attendance_by_date and roll_no in attendance_by_date[date_str]:
                                                status = attendance_by_date[date_str][roll_no]
                                                row[date_str] = "✅" if status == "present" else "❌"
                                                total_count += 1
                                                if status == "present":
                                                    present_count += 1
                                            else:
                                                row[date_str] = ""
                                        
                                        # Add summary statistics
                                        if total_count > 0:
                                            row["Present"] = present_count
                                            row["Total"] = total_count
                                            row["Attendance %"] = round(present_count / total_count * 100, 1)
                                        else:
                                            row["Present"] = 0
                                            row["Total"] = 0
                                            row["Attendance %"] = 0.0
                                            
                                        rows.append(row)
                                    
                                    # Convert to DataFrame
                                    subject_df = pd.DataFrame(rows)
                                    
                                    # Display summary statistics
                                    avg_attendance = subject_df["Attendance %"].mean() if len(subject_df) > 0 else 0
                                    below_75_count = len(subject_df[subject_df["Attendance %"] < 75]) if len(subject_df) > 0 else 0
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Average Attendance", f"{avg_attendance:.1f}%")
                                    with col2:
                                        st.metric("Students Below 75%", str(below_75_count))
                                    with col3:
                                        st.metric("Total Classes", str(len(dates)))
                                    
                                    # Display attendance data
                                    summary_cols = ["Roll No", "Name", "Present", "Total", "Attendance %"]
                                    detail_cols = ["Roll No", "Name"] + dates + ["Attendance %"]
                                    
                                    # Create tabs for summary and detailed views
                                    view_tabs = st.tabs(["Summary", "Detailed"])
                                    
                                    with view_tabs[0]:
                                        # Display summary view
                                        styled_subject_df = subject_df[summary_cols].style.applymap(
                                            color_attendance, 
                                            subset=pd.IndexSlice[:, ['Attendance %']]
                                        )
                                        st.dataframe(styled_subject_df, use_container_width=True)
                                    
                                    with view_tabs[1]:
                                        # Display detailed view with dates
                                        st.dataframe(subject_df[detail_cols], use_container_width=True)
                                    
                                    # Export to Excel
                                    excel_path = os.path.join("excel_exports", f"{subject_name}_{department}_{year}{division}.xlsx")
                                    
                                    # Auto-update Excel file
                                    try:
                                        # First, check if the file exists and try to read it
                                        if os.path.exists(excel_path):
                                            try:
                                                existing_df = pd.read_excel(excel_path)
                                                
                                                # Get existing dates in the Excel file
                                                existing_dates = [col for col in existing_df.columns if col not in ["Roll No", "Name", "Present", "Total", "Attendance %"]]
                                                
                                                # Merge with new data
                                                for date_str in dates:
                                                    if date_str not in existing_dates:
                                                        # Add new date column
                                                        for _, row in subject_df.iterrows():
                                                            roll_no = row["Roll No"]
                                                            existing_df.loc[existing_df["Roll No"] == roll_no, date_str] = row.get(date_str, "")
                                                
                                                # Recalculate summary statistics
                                                for i, row in existing_df.iterrows():
                                                    present_count = sum(1 for col in existing_df.columns if col not in ["Roll No", "Name", "Present", "Total", "Attendance %"] and row[col] == "✅")
                                                    total_count = sum(1 for col in existing_df.columns if col not in ["Roll No", "Name", "Present", "Total", "Attendance %"] and row[col] in ["✅", "❌"])
                                                    
                                                    existing_df.at[i, "Present"] = present_count
                                                    existing_df.at[i, "Total"] = total_count
                                                    existing_df.at[i, "Attendance %"] = round(present_count / total_count * 100, 1) if total_count > 0 else 0.0
                                                
                                                # Save updated Excel file
                                                existing_df.to_excel(excel_path, index=False)
                                                st.success(f"Automatically updated {subject_name} Excel file with new attendance data.")
                                            except Exception as e:
                                                # If reading fails, just write a new file
                                                logger.error(f"Error reading existing Excel file: {str(e)}")
                                                subject_df.to_excel(excel_path, index=False)
                                        else:
                                            # Create new Excel file
                                            subject_df.to_excel(excel_path, index=False)
                                            
                                        # Provide download button
                                        with open(excel_path, "rb") as file:
                                            st.download_button(
                                                label=f"Download {subject_name} Excel Report",
                                                data=file,
                                                file_name=f"{subject_name}_Attendance.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                key=f"download_{subject_name}"
                                            )
                                    except Exception as excel_error:
                                        logger.error(f"Error with Excel operations: {str(excel_error)}")
                                        st.error(f"Error creating/updating Excel file: {str(excel_error)}")
                                else:
                                    st.info(f"No attendance records found for {subject_name} in the selected date range.")
                except Exception as e:
                    logger.error(f"Error displaying subject-wise attendance: {str(e)}")
                    st.error(f"Error displaying subject-wise attendance: {str(e)}")
                
                # Export overall class report to Excel
                excel_path = os.path.join("excel_exports", f"Class_Report_{department}_{year}{division}_{date_from}_to_{date_to}.xlsx")
                df_summary.to_excel(excel_path, index=False)
            else:
                st.info(f"No attendance data found for the selected date range ({date_from} to {date_to}).")
    
    except Exception as e:
        logger.error(f"Error generating class report: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error: {str(e)}")

elif page == "Student Reports":
    st.title("👨‍🎓 Student-wise Attendance Reports")
    st.markdown('<div class="dashboard-card"><p>View detailed attendance for individual students.</p></div>', unsafe_allow_html=True)
    
    try:
        # Get all students
        all_students = get_all_students()
        
        if not all_students:
            st.warning("No students found in database. Please register students first.")
        else:
            # Convert to DataFrame for selection
            students_df = pd.DataFrame([
                {"id": student["id"], "roll_no": student["roll_no"], "name": student["name"]}
                for student in all_students
            ])
            
            # Create a search box for students
            search_term = st.text_input("Search student by roll number or name:")
            
            if search_term:
                filtered_students = students_df[
                    students_df["roll_no"].str.contains(search_term, case=False) | 
                    students_df["name"].str.contains(search_term, case=False)
                ]
            else:
                filtered_students = students_df
            
            # Display students in a selectbox
            student_options = [f"{row['roll_no']} - {row['name']}" for _, row in filtered_students.iterrows()]
            
            if student_options:
                selected_student_option = st.selectbox("Select Student:", student_options)
                selected_roll_no = selected_student_option.split(" - ")[0]
                
                try:
                    # Get student ID
                    student_id = students_df[students_df["roll_no"] == selected_roll_no]["id"].values[0]
                    
                    # Get student details
                    student_details = get_student_details(student_id)
                    
                    if student_details is None:
                        st.error("Student details not found in database. The student may have been deleted.")
                    else:
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["Summary", "Detailed Report"])
                        
                        with tab1:
                            # Display student information
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                try:
                                    # Display student image
                                    img = Image.open(student_details["image_path"])
                                    st.image(img, width=150)
                                except:
                                    st.info("No image available")
                            
                            with col2:
                                st.markdown(f"### {student_details['name']}")
                                st.markdown(f"**Roll No:** {student_details['roll_no']}")
                                st.markdown(f"**Email:** {student_details['email'] or 'N/A'}")
                                st.markdown(f"**Department:** {student_details['department']} | **Year:** {student_details['year']} | **Division:** {student_details['division']}")
                            
                            # Get attendance summary
                            attendance_summary = get_student_attendance_summary(student_id)
                            
                            if attendance_summary:
                                # Convert to DataFrame
                                df_summary = pd.DataFrame([
                                    {
                                        "Subject Code": row["subject_code"],
                                        "Subject Name": row["subject_name"],
                                        "Present": row["present_count"],
                                        "Total Classes": row["total_classes"],
                                        "Attendance %": round(row["present_count"] / row["total_classes"] * 100, 1) if row["total_classes"] > 0 else 0
                                    }
                                    for row in attendance_summary
                                ])
                                
                                # Calculate overall attendance percentage
                                overall_present = df_summary["Present"].sum()
                                overall_total = df_summary["Total Classes"].sum()
                                overall_percentage = round(overall_present / overall_total * 100, 1) if overall_total > 0 else 0
                                
                                # Show overall stats
                                st.markdown("### Overall Attendance")
                                
                                # Determine status color based on attendance
                                status_color = "#4caf50" if overall_percentage >= 75 else "#ff9800" if overall_percentage >= 50 else "#f44336"
                                
                                # Create attendance gauge chart
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=overall_percentage,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Overall Attendance"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': status_color},
                                        'steps': [
                                            {'range': [0, 50], 'color': "#ffcdd2"},
                                            {'range': [50, 75], 'color': "#ffecb3"},
                                            {'range': [75, 100], 'color': "#c8e6c9"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 75
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=250)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show subject-wise attendance
                                st.markdown("### Subject-wise Attendance")
                                
                                # Create bar chart for subject-wise attendance
                                fig = px.bar(
                                    df_summary,
                                    x="Subject Code",
                                    y="Attendance %",
                                    title="Subject-wise Attendance Percentage",
                                    color="Attendance %",
                                    color_continuous_scale=["red", "orange", "green"],
                                    range_color=[0, 100],
                                    text="Attendance %"
                                )
                                
                                fig.update_layout(height=400)
                                fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Attendance Threshold (75%)")
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display the detailed table
                                st.markdown("### Subject Details")
                                
                                # Add color formatting based on attendance percentage
                                def color_attendance(val):
                                    if val < 50:
                                        return 'color: #f44336; font-weight: bold'  # Red
                                    elif val < 75:
                                        return 'color: #ff9800; font-weight: bold'  # Orange
                                    else:
                                        return 'color: #4caf50; font-weight: bold'  # Green
                                
                                # Apply the formatting to the attendance percentage column
                                styled_df = df_summary.style.applymap(
                                    color_attendance, 
                                    subset=pd.IndexSlice[:, ['Attendance %']]
                                )
                                
                                st.dataframe(styled_df, use_container_width=True)
                            else:
                                st.info("No attendance data found for this student.")
                        
                        with tab2:
                            # Get detailed attendance report
                            attendance_report = get_student_attendance_report(student_id)
                            
                            if attendance_report:
                                # Convert to DataFrame
                                df_report = pd.DataFrame([
                                    {
                                        "Date": row["date"],
                                        "Subject": f"{row['subject_code']} - {row['subject_name']}",
                                        "Period": row["period"],
                                        "Status": "✅" if row["status"] == "present" else "❌"
                                    }
                                    for row in attendance_report
                                ])
                                
                                # Convert date column to datetime
                                df_report["Date"] = pd.to_datetime(df_report["Date"])
                                
                                # Add day of week column
                                df_report["Day"] = df_report["Date"].dt.day_name()
                                
                                # Group by date to create calendar view
                                cal_data = df_report.groupby("Date")["Status"].agg(
                                    lambda x: "✅" if all(s == "✅" for s in x) else 
                                              "❌" if all(s == "❌" for s in x) else "⚠️"
                                )
                                
                                st.markdown("### Attendance Calendar")
                                
                                # Create a monthly view
                                months = sorted(df_report["Date"].dt.to_period("M").unique())
                                
                                for month_period in months:
                                    month_start = month_period.start_time
                                    month_name = month_start.strftime("%B %Y")
                                    
                                    st.markdown(f"#### {month_name}")
                                    
                                    # Filter data for this month
                                    month_data = cal_data[cal_data.index.to_period("M") == month_period]
                                    
                                    # Create a calendar for this month
                                    _, num_days = calendar.monthrange(month_start.year, month_start.month)
                                    
                                    # Create a 7x6 calendar layout (7 days per week, up to 6 weeks)
                                    calendar_data = []
                                    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                                    
                                    # Create week rows
                                    first_day_weekday = month_start.replace(day=1).weekday()  # 0 is Monday
                                    day_count = 1
                                    
                                    for week in range(6):  # Max 6 weeks in a month
                                        week_data = []
                                        for weekday in range(7):  # 7 days in a week
                                            if (week == 0 and weekday < first_day_weekday) or day_count > num_days:
                                                # Empty cell
                                                week_data.append("")
                                            else:
                                                date_str = f"{month_start.year}-{month_start.month:02d}-{day_count:02d}"
                                                date = pd.Timestamp(date_str)
                                                
                                                if date in month_data.index:
                                                    week_data.append(f"{day_count} {month_data[date]}")
                                                else:
                                                    week_data.append(f"{day_count}")
                                                
                                                day_count += 1
                                        
                                        calendar_data.append(week_data)
                                    
                                    # Only include weeks that have days
                                    calendar_data = [week for week in calendar_data if any(cell != "" for cell in week)]
                                    
                                    # Create DataFrame for display
                                    calendar_df = pd.DataFrame(calendar_data, columns=day_names)
                                    
                                    # Display as a styled table
                                    def style_calendar(val):
                                        if "✅" in val:
                                            return 'background-color: #c8e6c9; font-weight: bold'
                                        elif "❌" in val:
                                            return 'background-color: #ffcdd2; font-weight: bold'
                                        elif "⚠️" in val:
                                            return 'background-color: #fff9c4; font-weight: bold'
                                        else:
                                            return ''
                                    
                                    # Apply the styling
                                    styled_calendar = calendar_df.style.applymap(style_calendar)
                                    st.dataframe(styled_calendar, use_container_width=True, hide_index=True)
                                
                                # Detailed attendance list
                                st.markdown("### Detailed Attendance List")
                                
                                # Sort by date (newest first) and format date column
                                df_report = df_report.sort_values("Date", ascending=False)
                                df_report["Date"] = df_report["Date"].dt.strftime("%Y-%m-%d")
                                
                                # Display as a styled dataframe
                                def style_status(val):
                                    if val == "✅":
                                        return 'color: #4caf50; font-weight: bold'
                                    else:
                                        return 'color: #f44336; font-weight: bold'
                                
                                # Apply the styling
                                styled_df = df_report.style.applymap(
                                    style_status, 
                                    subset=pd.IndexSlice[:, ['Status']]
                                )
                                
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                                
                                # Export to Excel
                                excel_path = os.path.join("excel_exports", f"Student_Report_{selected_roll_no}.xlsx")
                                df_report.to_excel(excel_path, index=False)
                                
                                with open(excel_path, "rb") as file:
                                    st.download_button(
                                        label="Download Excel Report",
                                        data=file,
                                        file_name=f"Attendance_Report_{selected_roll_no}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            else:
                                st.info("No detailed attendance records found for this student.")
                                
                except IndexError:
                    logger.error(f"Student with roll number {selected_roll_no} not found in DataFrame")
                    st.error(f"Student with roll number {selected_roll_no} not found in the student database. This may happen if the student record was deleted after the page loaded.")
                except Exception as e:
                    logger.error(f"Error retrieving student details: {str(e)}")
                    st.error(f"Failed to retrieve student details: {str(e)}")
            else:
                st.warning("No students match your search criteria.")
    
    except Exception as e:
        logger.error(f"Error generating student report: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error: {str(e)}")

elif page == "Edit Attendance":
    st.title("🔄 Edit Attendance")
    st.markdown('<div class="dashboard-card"><p>View and edit previous attendance records.</p></div>', unsafe_allow_html=True)
    
    # Create columns for form inputs
    col1, col2, col3 = st.columns(3)
    
    # Get subjects from database with error handling
    try:
        subjects = get_subjects()
        subject_options = [subject[1] for subject in subjects]  # Use code instead of full name
        
        with col1:
            if subject_options:
                selected_subject = st.selectbox("Select Subject", subject_options)
                subject_id = next((subject[0] for subject in subjects if subject[1] == selected_subject), None)
            else:
                st.warning("No subjects found in database. Please check database configuration.")
                selected_subject = None
                subject_id = None
        
        with col2:
            # Date selection
            report_date = st.date_input("Select Date", value=datetime.date.today())
        
        with col3:
            periods = [
                "10:15 - 11:15",
                "11:15 - 12:15",
                "01:15 - 02:15",
                "02:15 - 03:15",
                "03:30 - 04:30",
                "04:30 - 05:30"
            ]
            selected_period = st.selectbox("Select Period", periods)
        
        # Load attendance data if all parameters are selected
        if subject_id:
            # Get attendance data
            attendance_data = get_attendance_report(subject_id, report_date.strftime("%Y-%m-%d"))
            
            if attendance_data:
                st.subheader(f"Attendance for {selected_subject} on {report_date}")
                
                # Convert to dataframe and add checkbox for editing
                attendance_df = pd.DataFrame(
                    [(data[0], data[1], data[2], data[3], data[4]) for data in attendance_data],
                    columns=["ID", "Roll No", "Name", "Email", "Status"]
                )
                
                # Display attendance statistics
                present_count = sum(attendance_df["Status"] == "present")
                absent_count = sum(attendance_df["Status"] == "absent")
                total_count = len(attendance_df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Students", str(total_count))
                with col2:
                    st.metric("Present", str(present_count))
                with col3:
                    st.metric("Absent", str(absent_count))
                with col4:
                    attendance_percentage = (present_count / total_count * 100) if total_count > 0 else 0
                    st.metric("Attendance %", f"{attendance_percentage:.1f}%")
                
                # Create a form to allow editing
                with st.form(key=f"edit_attendance_{subject_id}_{report_date}"):
                    st.markdown("### Edit Attendance Status")
                    st.markdown("Check the boxes for students who are **present**, uncheck for **absent** students.")
                    
                    # Create attendance edit options
                    attendance_status = {}
                    
                    for _, row in attendance_df.iterrows():
                        student_id = row["ID"]
                        roll_no = row["Roll No"]
                        name = row["Name"]
                        is_present = row["Status"] == "present"
                        
                        attendance_status[student_id] = st.checkbox(
                            f"{roll_no} - {name}",
                            value=is_present,
                            key=f"attendance_{student_id}"
                        )
                    
                    # Submit button
                    submit = st.form_submit_button("Update Attendance")
                    
                    if submit:
                        conn = get_connection()
                        cursor = conn.cursor()
                        
                        try:
                            update_count = 0
                            
                            for student_id, is_present in attendance_status.items():
                                status = "present" if is_present else "absent"
                                
                                # Check if we need to update or insert
                                cursor.execute('''
                                    SELECT id FROM attendance 
                                    WHERE student_id = ? AND subject_id = ? AND date = ? AND period = ?
                                ''', (student_id, subject_id, report_date.strftime("%Y-%m-%d"), selected_period))
                                
                                existing = cursor.fetchone()
                                
                                if existing:
                                    # Update existing record
                                    cursor.execute('''
                                        UPDATE attendance
                                        SET status = ?
                                        WHERE student_id = ? AND subject_id = ? AND date = ? AND period = ?
                                    ''', (status, student_id, subject_id, report_date.strftime("%Y-%m-%d"), selected_period))
                                    
                                    if cursor.rowcount > 0:
                                        update_count += 1
                                else:
                                    # Insert new record
                                    cursor.execute('''
                                        INSERT INTO attendance (student_id, subject_id, date, period, status)
                                        VALUES (?, ?, ?, ?, ?)
                                    ''', (student_id, subject_id, report_date.strftime("%Y-%m-%d"), selected_period, status))
                                    
                                    if cursor.rowcount > 0:
                                        update_count += 1
                            
                            conn.commit()
                            st.success(f"Successfully updated attendance for {update_count} students!")
                            
                            # Add button to refresh the page
                            st.button("Refresh", key="refresh_attendance")
                            
                        except Exception as e:
                            conn.rollback()
                            logger.error(f"Error updating attendance: {str(e)}")
                            st.error(f"Failed to update attendance: {str(e)}")
                        finally:
                            conn.close()
                
            else:
                st.info(f"No attendance records found for {selected_subject} on {report_date}. You can create a new attendance record.")
                
                # Allow creating new attendance record manually
                with st.form(key=f"new_attendance_{subject_id}_{report_date}"):
                    st.markdown("### Create New Attendance Record")
                    st.markdown("Check the boxes for students who are **present**.")
                    
                    # Get all students enrolled in the subject
                    all_students = get_students_by_subject(subject_id)
                    
                    if not all_students:
                        st.warning("No students enrolled in this subject.")
                    else:
                        # Create attendance options
                        attendance_status = {}
                        
                        for student in all_students:
                            student_id = student["id"]
                            roll_no = student["roll_no"]
                            name = student["name"]
                            
                            attendance_status[student_id] = st.checkbox(
                                f"{roll_no} - {name}",
                                value=False,
                                key=f"new_attendance_{student_id}"
                            )
                        
                        # Submit button
                        submit = st.form_submit_button("Save Attendance")
                        
                        if submit:
                            try:
                                success_count = 0
                                
                                for student_id, is_present in attendance_status.items():
                                    if is_present:
                                        mark_attendance(student_id, subject_id, report_date.strftime("%Y-%m-%d"), selected_period)
                                        success_count += 1
                                
                                st.success(f"Successfully created attendance record with {success_count} present students!")
                                
                                # Add button to refresh the page
                                st.button("Refresh", key="refresh_new_attendance")
                                
                            except Exception as e:
                                logger.error(f"Error creating attendance: {str(e)}")
                                st.error(f"Failed to create attendance: {str(e)}")
        else:
            st.info("Please select a subject, date, and period to view or edit attendance.")
    
    except Exception as e:
        logger.error(f"Error in Edit Attendance page: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error: {str(e)}")

# Footer
st.sidebar.divider()
st.sidebar.info(
    "Automated Facial Attendance System\n"
    "Developed for ENTC TY B"
) 