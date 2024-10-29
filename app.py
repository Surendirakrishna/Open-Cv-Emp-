from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
import os
from datetime import date, datetime
import xlrd
from xlutils.copy import copy as xl_copy
from xlwt import Workbook

app = Flask(__name__)

def load_and_encode_image(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return None
    return encodings[0]

def initialize_excel_file(excel_file, lecture_name):
    if not os.path.exists(excel_file):
        wb = Workbook()
        sheet = wb.add_sheet(lecture_name)
        sheet.write(0, 0, 'Name')
        sheet.write(0, 1, 'Date')
        sheet.write(0, 2, 'Time')
        sheet.write(0, 3, 'Status')
        wb.save(excel_file)
    rb = xlrd.open_workbook(excel_file, formatting_info=True)
    wb = xl_copy(rb)
    sheet_names = rb.sheet_names()
    if lecture_name in sheet_names:
        sheet = wb.get_sheet(sheet_names.index(lecture_name))
    else:
        sheet = wb.add_sheet(lecture_name)
        sheet.write(0, 0, 'Name')
        sheet.write(0, 1, 'Date')
        sheet.write(0, 2, 'Time')
        sheet.write(0, 3, 'Status')
        wb.save(excel_file)
    return wb, sheet, rb

def save_unknown_face(image, face_location, save_path):
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    cv2.imwrite(save_path, face_image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    lecture_name = request.form['lecture_name']
    current_folder = os.getcwd()
    unknown_faces_folder = os.path.join(current_folder, "unknown_faces")
    os.makedirs(unknown_faces_folder, exist_ok=True)

    image_files = {
        'suren': os.path.join(current_folder, 'suren.png'),
        'aakash_s': os.path.join(current_folder, 'aakash_s.png'),
        'govarthan': os.path.join(current_folder, 'govarthan.png')
    }

    known_face_encodings = []
    known_face_names = []
    for name, path in image_files.items():
        if not os.path.exists(path):
            return f"Image file not found: {path}"
        encoding = load_and_encode_image(path)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return "Cannot open webcam"

    excel_file = 'attendance_excel.xls'
    wb, sheet, rb = initialize_excel_file(excel_file, lecture_name)
    row = rb.sheet_by_name(lecture_name).nrows
    already_attendance_taken = set()

    face_recognition_threshold = 0.6

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = "Unknown"

                if face_distances[best_match_index] < face_recognition_threshold:
                    name = known_face_names[best_match_index]

                if name != "Unknown" and name not in already_attendance_taken:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    status = "Present" if datetime.now().time() < datetime.strptime("09:00:00", "%H:%M:%S").time() else "Absent"
                    
                    sheet.write(row, 0, name)
                    sheet.write(row, 1, str(date.today()))
                    sheet.write(row, 2, current_time)
                    sheet.write(row, 3, status)
                    wb.save(excel_file)
                    already_attendance_taken.add(name)
                    row += 1
                elif name == "Unknown":
                    unknown_face_filename = os.path.join(unknown_faces_folder, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    save_unknown_face(frame, face_location, unknown_face_filename)
        video_capture.release()
        wb.save(excel_file)
        return render_template('result.html', result="Attendance has been taken successfully.")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
