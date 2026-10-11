import cv2
import os

# Model file paths
model_path = r"C:\kavya project 2-1\res10_300x300_ssd_iter_140000.caffemodel"
config_path = r"C:\kavya project 2-1\deploy.prototxt"

# Load face detection model
if not os.path.exists(model_path) or not os.path.exists(config_path):
    print("Error: Model files not found. Check the file paths.")
    raise SystemExit

net = cv2.dnn.readNetFromCaffe(config_path, model_path)


# Function to detect faces
def detect_faces(frame):
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()
    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [
                width, height, width, height
            ]

            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            if x2 > x1 and y2 > y1:
                face_count += 1

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Face: {confidence * 100:.1f}%",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

    cv2.putText(
        frame, f"Faces detected: {face_count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2
    )

    return frame


# Main menu
while True:
    print("\n===== FACE DETECTION PROJECT =====")
    print("1. Detect faces in an image")
    print("2. Detect faces in a video")
    print("3. Detect faces using webcam")
    print("4. Exit")

    choice = input("Enter your choice: ")

    # Image detection
    if choice == "1":
        image_path = input("Enter image path: ").strip().strip('"')
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not open image.")
            continue

        result = detect_faces(image)
        cv2.imshow("Image Face Detection", result)
        cv2.imwrite("detected_image.jpg", result)
        print("Result saved as detected_image.jpg")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Video file detection
    elif choice == "2":
        video_path = input("Enter video path: ").strip().strip('"')
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            continue

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            result = detect_faces(frame)
            cv2.imshow("Video Face Detection", result)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Live webcam detection
    elif choice == "3":
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            continue

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Unable to read webcam.")
                break

            result = detect_faces(frame)
            cv2.imshow("Live Face Detection", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif choice == "4":
        print("Face Detection Project Closed.")
        break

    else:
        print("Invalid choice. Enter 1, 2, 3, or 4.")
