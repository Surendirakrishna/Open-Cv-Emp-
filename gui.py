import cv2
import face_recognition
import numpy as np
import os
from datetime import date, datetime
import xlrd
from xlutils.copy import copy as xl_copy
from xlwt import Workbook
import tkinter as tk
from tkinter import messagebox, simpledialog

def load_and_encode_image(image_path):
    try: 
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            raise ValueError(f"No face found in image: {image_path}")
        return encodings[0]
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")

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
        print(f"Sheet '{lecture_name}' already exists. Using existing sheet.")
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

def start_attendance_system():
    current_folder = os.getcwd()
    unknown_faces_folder = os.path.join(current_folder, "unknown_faces")
    os.makedirs(unknown_faces_folder, exist_ok=True)

    image_files = {
        'suren': os.path.join(current_folder, 'suren.png'),
        'sriharan': os.path.join(current_folder, 'sriharan.png'),
        'govarthan': os.path.join(current_folder, 'govarthan.png')
    }

    known_face_encodings = []
    known_face_names = []
    for name, path in image_files.items():
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Image file not found: {path}")
            return
        encoding = load_and_encode_image(path)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
        print(f"Loaded and encoded image for {name}")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        messagebox.showerror("Error", "Cannot open webcam")
        return

    lecture_name = simpledialog.askstring("Input", "Please enter the current subject lecture name:")
    if not lecture_name:
        return

    excel_file = 'attendance_excel.xls'
    wb, sheet, rb = initialize_excel_file(excel_file, lecture_name)
    row = rb.sheet_by_name(lecture_name).nrows
    already_attendance_taken = set()

    process_this_frame = True
    face_recognition_threshold = 0.6

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = "Unknown"

                    if face_distances[best_match_index] < face_recognition_threshold:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                    if name != "Unknown":
                        if name not in already_attendance_taken:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            status = "Present" if datetime.now().time() < datetime.strptime("09:00:00", "%H:%M:%S").time() else "Absent"
                            
                            sheet.write(row, 0, name)
                            sheet.write(row, 1, str(date.today()))
                            sheet.write(row, 2, current_time)
                            sheet.write(row, 3, status)
                            wb.save(excel_file)
                            already_attendance_taken.add(name)
                            row += 1
                            print(f"Attendance taken for {name} at {current_time} - Status: {status}")
                        else:
                            continue  # Skip further processing for this person if attendance is already taken
                    else:
                        print("Unknown face detected")
                        unknown_face_filename = os.path.join(unknown_faces_folder, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        save_unknown_face(frame, face_location, unknown_face_filename)
                        print(f"Saved unknown face to {unknown_face_filename}")

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting and saving data...")
                break
    finally:
        video_capture.release()
        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Video')
        wb.save(excel_file)
        print("Attendance has been saved successfully.")

def create_gui():
    root = tk.Tk()
    root.title("Smart Attendance System")

    start_button = tk.Button(root, text="Start Attendance", command=start_attendance_system)
    start_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
