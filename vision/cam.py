import cv2
import mediapipe as mp
import pyautogui
import math

# Global flag to stop the hand mouse
stop_hand = False

def virtual_hand_mouse():
    global stop_hand
    stop_hand = False

    screen_width, screen_height = pyautogui.size()

    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Camera not opened")
        return

    # Mediapipe hands
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    drawing_utils = mp.solutions.drawing_utils

    prev_x, prev_y = 0, 0
    smoothening = 7
    click_threshold = 40
    scroll_threshold = 60

    # Create resizable window
    cv2.namedWindow("Virtual Hand Mouse", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Hand Mouse", 1280, 720)

    print("Virtual hand mouse running. Press 'q' to quit.")

    while not stop_hand:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror image so it feels natural
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hand_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks lightly so face is visible
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2)
            )

            # Finger positions
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

            x = int(thumb_tip.x * screen_width)
            y = int(thumb_tip.y * screen_height)

            cur_x = prev_x + (x - prev_x) / smoothening
            cur_y = prev_y + (y - prev_y) / smoothening
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # Left click
            distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y) * screen_width
            if distance < click_threshold:
                pyautogui.click()
            
            # Scroll
            scroll_dist = math.hypot(middle_tip.x - index_tip.x, middle_tip.y - index_tip.y) * screen_width
            if scroll_dist < scroll_threshold:
                pyautogui.scroll(20)

        # Show frame
        cv2.imshow("Virtual Hand Mouse", frame)

        # Wait 1ms & check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_hand = True
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Virtual hand mouse stopped.")


# Test CLI
if __name__ == "__main__":
    virtual_hand_mouse()