import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import time

st.set_page_config(page_title="Real-Time Face ID", layout="centered")
st.title("Real-Time Face ID")

known_faces_dir = "known_faces"

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])


# Take Photo Button (appears only when toggle is on)
if "face_image" not in st.session_state:
    st.session_state["face_image"] = None


run = st.toggle("Start Face Identification")
frame_display = st.empty()


if st.button("Take a photo to add to database"):
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to access webcam.")
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)

        if len(face_locations) == 0:
            st.error("No face detected.")
        else:
            face_location = max(face_locations, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))
            top, right, bottom, left = face_location
            st.session_state["face_image"] = rgb[top:bottom, left:right]


if st.session_state["face_image"] is not None:
    with st.form("save_face_form"):
        st.image(st.session_state["face_image"], caption="Detected face preview", use_container_width=True)
        name = st.text_input("Enter name to save as:")
        submitted = st.form_submit_button("Confirm and Save")

        if submitted and name:
            save_path = os.path.join(known_faces_dir, f"{name}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(st.session_state["face_image"], cv2.COLOR_RGB2BGR))
            st.success(f"Saved {name}.jpg to database.")
            st.session_state["face_image"] = None
            st.rerun()


if run:
    # Real-time face recognition loop
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()

        if not ret:
            st.warning("Failed to access webcam.")
            break

        # Resize frame for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Face detection
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        scaled_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

        for (top, right, bottom, left), face_encoding in zip(scaled_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(rgb_frame)

    cap.release()


with st.expander("Upload a Picture to Add to Database"):
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)

        if len(face_locations) == 0:
            st.error("No face detected in the uploaded image.")
        else:
            face_location = max(
                face_locations,
                key=lambda box: (box[2] - box[0]) * (box[1] - box[3])
            )
            top, right, bottom, left = face_location
            face_image = rgb[top:bottom, left:right]

            st.image(face_image, caption="Detected face from upload", use_container_width=True)

            name = st.text_input("Enter name to save uploaded face as:", key="upload_name")
            if name and st.button("Save Uploaded Face"):
                save_path = os.path.join(known_faces_dir, f"{name}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                st.success(f"Saved {name}.jpg to database.")


with st.expander("Database Management"):
    face_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not face_files:
        st.info("No entries found in the database.")
    else:
        for filename in face_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(filename)
            with col2:
                if st.button(f"Delete", key=f"del_{filename}"):
                    os.remove(os.path.join(known_faces_dir, filename))
                    st.success(f"Deleted {filename}")
                    st.rerun()

        if st.button("Clear Entire Database"):
            for filename in face_files:
                os.remove(os.path.join(known_faces_dir, filename))
            st.success("All entries deleted.")
            st.rerun()
