import time
import config


class DrowsinessDetector:

    def __init__(self):

        self.closed_eye_counter = 0
        self.yawning_counter = 0

        self.last_alarm = 0

    def update(self, ear, mar):

        drowsy = False
        yawning = False

        # -------------------------
        # Eye detection
        # -------------------------

        if ear < config.EAR_THRESHOLD:

            self.closed_eye_counter += 1

        else:

            self.closed_eye_counter = 0

        if self.closed_eye_counter >= config.CLOSED_EYE_FRAMES:

            drowsy = True

        # -------------------------
        # Yawning detection
        # -------------------------

        if mar > config.MAR_THRESHOLD:

            self.yawning_counter += 1

        else:

            self.yawning_counter = 0

        if self.yawning_counter >= config.YAWNING_FRAMES:

            yawning = True

        return {
            "drowsy": drowsy,
            "yawning": yawning,
            "closed_eye_frames": self.closed_eye_counter,
            "yawning_frames": self.yawning_counter
        }

    def should_alarm(self):

        current_time = time.time()

        if current_time - self.last_alarm > config.ALARM_COOLDOWN:

            self.last_alarm = current_time

            return True

        return False