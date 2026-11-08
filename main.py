import cv2
import time

from face_features import FaceFeatures
from drowsiness_detector import DrowsinessDetector

import config


def main():

    camera = cv2.VideoCapture(config.CAMERA_ID)

    if not camera.isOpened():
        print("Could not open camera")
        return

    face = FaceFeatures()
    detector = DrowsinessDetector()

    previous_time = time.time()

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        points = face.process(frame)

        status = "ALERT"

        if points is not None:

            features = face.get_features(points)

            ear = features["EAR"]
            mar = features["MAR"]

            result = detector.update(ear, mar)

            # Draw landmarks
            frame = face.draw_landmarks(frame, points)

            # -------------------------
            # Status
            # -------------------------

            if result["drowsy"]:

                status = "DROWSY"

                cv2.putText(
                    frame,
                    "WAKE UP!",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4
                )

                if detector.should_alarm():

                    print("\a")

            elif result["yawning"]:

                status = "YAWNING"

            # Display EAR / MAR
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:

            status = "NO FACE"

        # Status
        cv2.putText(
            frame,
            f"STATUS: {status}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if status == "DROWSY" else (0, 255, 0),
            3
        )

        # FPS
        current_time = time.time()

        fps = 1 / (current_time - previous_time)

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Driver Drowsiness Detection", frame)

        # ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()