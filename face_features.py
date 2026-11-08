import cv2
import numpy as np
import mediapipe as mp


class FaceFeatures:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        self.MOUTH = [
            61,   # left
            291,  # right
            13,   # top
            14    # bottom
        ]

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        h, w = frame.shape[:2]

        points = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in landmarks
        ])

        return points

    def eye_aspect_ratio(self, eye):

        p1, p2, p3, p4, p5, p6 = eye

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)

        horizontal = np.linalg.norm(p1 - p4)

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return ear

    def mouth_aspect_ratio(self, points):

        left = points[self.MOUTH[0]]
        right = points[self.MOUTH[1]]

        top = points[self.MOUTH[2]]
        bottom = points[self.MOUTH[3]]

        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)

        return vertical / horizontal

    def get_features(self, points):

        left_eye = points[self.LEFT_EYE]
        right_eye = points[self.RIGHT_EYE]

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2

        mar = self.mouth_aspect_ratio(points)

        return {
            "EAR": ear,
            "MAR": mar
        }

    def draw_landmarks(self, frame, points):

        for point in points:
            cv2.circle(
                frame,
                tuple(point),
                1,
                (0, 255, 0),
                -1
            )

        return frame