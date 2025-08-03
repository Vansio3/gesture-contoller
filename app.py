import cv2
import mediapipe as mp
import pyautogui
import math
import time
from one_euro_filter import OneEuroFilter
import numpy as np
from collections import deque

pyautogui.FAILSAFE = False

# --- (Initialization and variables) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# --- Gesture area mapping ---
# Horizontal mapping 
HORIZONTAL_INPUT_START = 0.5  # Corresponds to the left edge of the screen
HORIZONTAL_INPUT_END = 0.9    # Corresponds to the right edge of the screen

# Vertical mapping 
VERTICAL_TOP_MARGIN = 0.5
VERTICAL_BOTTOM_MARGIN = 0.9


# --- Filter and precision settings ---
config = {'min_cutoff': 0.015, 'beta': 2.0}
x_filter = OneEuroFilter(time.time(), 0, min_cutoff=config['min_cutoff'], beta=config['beta'])
y_filter = OneEuroFilter(time.time(), 0, min_cutoff=config['min_cutoff'], beta=config['beta'])
DEAD_ZONE_RADIUS = 3.0
AVERAGING_WINDOW = 2
x_history = deque(maxlen=AVERAGING_WINDOW)
y_history = deque(maxlen=AVERAGING_WINDOW)


# --- Right-Hand Click Logic Variables ---
last_mouse_x, last_mouse_y = 0, 0
CLICK_THRESHOLD = 20
CLICK_COLOR_IDLE = (0, 255, 0)
CLICK_COLOR_PRESSED = (0, 0, 255)
CLICK_COLOR_COOLDOWN = (0, 0, 255)
CLICK_DISABLE_MOVE_THRESHOLD = 7.0
CLICK_COLOR_DISABLED = (0, 165, 255)
is_clicking = False
CLICK_COOLDOWN = 2.0
right_hand_detected_time = None

# --- Left-Hand Scroll Logic Variables ---
SCROLL_THRESHOLD = 22
SCROLL_COLOR_IDLE = (255, 255, 0)
SCROLL_COLOR_ACTIVE = (255, 0, 255)
SCROLL_COLOR_ANCHOR = (0, 255, 255)
scroll_start_y = None
SCROLL_DEAD_ZONE = 20
SCROLL_SENSITIVITY = 0.5

# --- Voice Typing Logic Variables ---
VOICE_TYPING_THRESHOLD = 25
VOICE_TYPING_COOLDOWN = 3.0
last_voice_typing_toggle_time = 0
VOICE_TYPING_COLOR_IDLE = (255, 255, 255)
VOICE_TYPING_COLOR_ACTIVE = (0, 255, 255)

# --- Ctrl+Enter Logic Variables ---
CTRL_ENTER_THRESHOLD = 25
CTRL_ENTER_COOLDOWN = 3.0
last_ctrl_enter_time = 0
CTRL_ENTER_COLOR_IDLE = (200, 0, 200)
CTRL_ENTER_COLOR_ACTIVE = (255, 0, 255)

# --- Pause Control Logic Variables ---
is_paused = False
PAUSE_COOLDOWN = 2.0
last_pause_toggle_time = 0
PAUSE_HOLD_DURATION = 2.0
pause_gesture_start_time = None

def are_all_fingers_extended(hand_landmarks):
    if not hand_landmarks:
        return False
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    for finger_tip_id, finger_pip_id in [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        tip = hand_landmarks.landmark[finger_tip_id]
        pip = hand_landmarks.landmark[finger_pip_id]
        dist_tip_wrist = math.dist((tip.x, tip.y), (wrist.x, wrist.y))
        dist_pip_wrist = math.dist((pip.x, pip.y), (wrist.x, wrist.y))
        if dist_tip_wrist < dist_pip_wrist:
            return False
    return True

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    current_time = time.time()
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    right_hand_landmarks = None
    left_hand_landmarks = None
    is_right_hand_present_this_frame = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks_iter in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness == "Right":
                right_hand_landmarks = hand_landmarks_iter
            elif handedness == "Left":
                left_hand_landmarks = hand_landmarks_iter

    # --- PAUSE/UNPAUSE LOGIC ---
    is_pause_gesture_active = False
    if left_hand_landmarks:
        wrist_x = left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        pinky_mcp_x = left_hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
        is_back_of_hand = pinky_mcp_x > wrist_x
        if is_back_of_hand and are_all_fingers_extended(left_hand_landmarks):
            is_pause_gesture_active = True

    can_initiate_pause = (current_time - last_pause_toggle_time) > PAUSE_COOLDOWN
    if is_pause_gesture_active and can_initiate_pause:
        if pause_gesture_start_time is None:
            pause_gesture_start_time = current_time

        elapsed_hold_time = current_time - pause_gesture_start_time
        
        wrist_pixel = (int(left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width),
                       int(left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height))
        progress = min(elapsed_hold_time / PAUSE_HOLD_DURATION, 1.0)
        cv2.ellipse(image, wrist_pixel, (30, 30), -90, 0, 360 * progress, (0, 255, 255), 4)
        status_text = "Pausing..." if not is_paused else "Unpausing..."
        cv2.putText(image, status_text, (wrist_pixel[0] - 60, wrist_pixel[1] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if elapsed_hold_time >= PAUSE_HOLD_DURATION:
            is_paused = not is_paused
            last_pause_toggle_time = current_time
            pause_gesture_start_time = None
    else:
        pause_gesture_start_time = None

    if is_paused:
        cv2.putText(image, "PAUSED", (image_width // 2 - 100, image_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('Hand Mouse Control', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue

    # --- TWO-HAND GESTURE LOGIC (Only runs if not paused) ---
    if right_hand_landmarks and left_hand_landmarks:
        right_index_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        right_index_pixel = (int(right_index_tip.x * image_width), int(right_index_tip.y * image_height))

        left_pinky_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        left_pinky_pixel = (int(left_pinky_tip.x * image_width), int(left_pinky_tip.y * image_height))
        voice_dist = math.dist(right_index_pixel, left_pinky_pixel)
        voice_line_color = VOICE_TYPING_COLOR_IDLE
        if voice_dist < VOICE_TYPING_THRESHOLD:
            if current_time - last_voice_typing_toggle_time > VOICE_TYPING_COOLDOWN:
                pyautogui.hotkey('win', 'h')
                last_voice_typing_toggle_time = current_time
            voice_line_color = VOICE_TYPING_COLOR_ACTIVE
        cv2.line(image, right_index_pixel, left_pinky_pixel, voice_line_color, 2)

        left_thumb_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        left_thumb_pixel = (int(left_thumb_tip.x * image_width), int(left_thumb_tip.y * image_height))
        ctrl_enter_dist = math.dist(right_index_pixel, left_thumb_pixel)
        ctrl_enter_line_color = CTRL_ENTER_COLOR_IDLE
        if ctrl_enter_dist < CTRL_ENTER_THRESHOLD:
            if current_time - last_ctrl_enter_time > CTRL_ENTER_COOLDOWN:
                pyautogui.hotkey('ctrl', 'enter')
                last_ctrl_enter_time = current_time
            ctrl_enter_line_color = CTRL_ENTER_COLOR_ACTIVE
        cv2.line(image, right_index_pixel, left_thumb_pixel, ctrl_enter_line_color, 2)

    # --- RIGHT HAND LOGIC (Cursor and Click - Only runs if not paused) ---
    if right_hand_landmarks:
        is_right_hand_present_this_frame = True
        if right_hand_detected_time is None: right_hand_detected_time = current_time
        
        wrist_landmark = right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_finger_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        # --- MODIFIED: Use new horizontal variables for interpolation ---
        raw_target_x = np.interp(wrist_landmark.x, [HORIZONTAL_INPUT_START, HORIZONTAL_INPUT_END], [0, screen_width])
        raw_target_y = np.interp(wrist_landmark.y, [VERTICAL_TOP_MARGIN, VERTICAL_BOTTOM_MARGIN], [0, screen_height])
        
        x_history.append(raw_target_x); y_history.append(raw_target_y)
        avg_x = sum(x_history) / len(x_history); avg_y = sum(y_history) / len(y_history)
        filtered_x = x_filter(current_time, avg_x); filtered_y = y_filter(current_time, avg_y)

        move_distance = math.dist((filtered_x, filtered_y), (last_mouse_x, last_mouse_y))
        if move_distance > DEAD_ZONE_RADIUS:
            pyautogui.moveTo(filtered_x, filtered_y)
            last_mouse_x, last_mouse_y = filtered_x, filtered_y

        index_pip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = right_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        is_index_extended = math.dist((wrist_landmark.x, wrist_landmark.y), (index_finger_tip.x, index_finger_tip.y)) > math.dist((wrist_landmark.x, wrist_landmark.y), (index_pip.x, index_pip.y))
        is_middle_extended = math.dist((wrist_landmark.x, wrist_landmark.y), (middle_tip.x, middle_tip.y)) > math.dist((wrist_landmark.x, wrist_landmark.y), (middle_pip.x, middle_pip.y))
        is_gesture_intentional = is_index_extended and is_middle_extended

        is_on_cooldown = (current_time - right_hand_detected_time) < CLICK_COOLDOWN
        is_moving_too_fast = move_distance > CLICK_DISABLE_MOVE_THRESHOLD
        
        index_pixel_x = int(index_finger_tip.x * image_width); index_pixel_y = int(index_finger_tip.y * image_height)
        thumb_pixel_x = int(thumb_tip.x * image_width); thumb_pixel_y = int(thumb_tip.y * image_height)
        click_distance = math.dist((thumb_pixel_x, thumb_pixel_y), (index_pixel_x, index_pixel_y))
        
        line_color = CLICK_COLOR_IDLE
        if click_distance < CLICK_THRESHOLD:
            if is_gesture_intentional and not is_on_cooldown and not is_moving_too_fast:
                line_color = CLICK_COLOR_PRESSED
                if not is_clicking: pyautogui.click(); is_clicking = True
            else:
                line_color = CLICK_COLOR_DISABLED; is_clicking = False
        else:
            is_clicking = False
        
        cv2.line(image, (thumb_pixel_x, thumb_pixel_y), (index_pixel_x, index_pixel_y), line_color, 2)
        wrist_pixel_x = int(wrist_landmark.x * image_width); wrist_pixel_y = int(wrist_landmark.y * image_height)
        control_point_color = CLICK_COLOR_IDLE if not is_on_cooldown else CLICK_COLOR_COOLDOWN
        cv2.circle(image, (wrist_pixel_x, wrist_pixel_y), 10, control_point_color, -1)
        mp_drawing.draw_landmarks(image, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- LEFT HAND LOGIC (Scroll - Only runs if not paused) ---
    if left_hand_landmarks:
        index_tip_left = left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip_left = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_pixel_x_left = int(index_tip_left.x * image_width); index_pixel_y_left = int(index_tip_left.y * image_height)
        thumb_pixel_x_left = int(thumb_tip_left.x * image_width); thumb_pixel_y_left = int(thumb_tip_left.y * image_height)
        
        scroll_distance = math.dist((thumb_pixel_x_left, thumb_pixel_y_left), (index_pixel_x_left, index_pixel_y_left))
        line_color_left = SCROLL_COLOR_IDLE
        
        if scroll_distance < SCROLL_THRESHOLD:
            line_color_left = SCROLL_COLOR_ACTIVE
            if scroll_start_y is None: scroll_start_y = index_pixel_y_left
            
            delta_y = scroll_start_y - index_pixel_y_left
            if abs(delta_y) > SCROLL_DEAD_ZONE:
                scroll_offset = delta_y - math.copysign(SCROLL_DEAD_ZONE, delta_y)
                pyautogui.scroll(int(scroll_offset * SCROLL_SENSITIVITY))
        else:
            scroll_start_y = None
            
        cv2.line(image, (thumb_pixel_x_left, thumb_pixel_y_left), (index_pixel_x_left, index_pixel_y_left), line_color_left, 2)
        if scroll_start_y is not None:
            cv2.circle(image, (index_pixel_x_left, scroll_start_y), 15, SCROLL_COLOR_ANCHOR, 2)
        mp_drawing.draw_landmarks(image, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if not is_right_hand_present_this_frame:
        right_hand_detected_time = None

    cv2.imshow('Hand Mouse Control', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()