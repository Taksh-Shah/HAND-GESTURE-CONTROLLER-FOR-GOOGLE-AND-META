import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import gc
import webbrowser

# Disable pyautogui failsafe for smoother operation
pyautogui.FAILSAFE = False

# Auto-launch YouTube Shorts
print("Opening YouTube Shorts...")
try:
    webbrowser.open('https://www.youtube.com/shorts')
    time.sleep(2)  # Give browser time to load
except Exception as e:
    print(f"Could not open YouTube Shorts: {e}")

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize camera with fallback
def initialize_camera():
    """Try multiple camera indices if default fails"""
    for i in range(3):  # Try camera indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} initialized successfully")
            return cap
        cap.release()
    raise Exception("No camera available")

# Initialize webcam with error handling
try:
    cap = initialize_camera()
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Increased for smoother tracking
    
except Exception as e:
    print(f"Camera initialization error: {e}")
    exit(1)

# Variables to track gesture state
initial_finger_vectors = {}
tracking_active = False
last_action_time = 0
gesture_lock_duration = 2.0  # 2 seconds gesture lock
index_angle_threshold = 13  # 13 degrees threshold for index finger
convergence_threshold = 0.08  # Reduced threshold for easier triggering
frame_count = 0
current_gesture_type = None  # Track current gesture type

# Performance optimization variables
no_hands_counter = 0
process_every_n_frames = 1

# Smoothing buffer for angle calculations
angle_buffer = []
buffer_size = 5

def smooth_angle(current_angle):
    """Smooth angle calculations using a buffer"""
    angle_buffer.append(current_angle)
    if len(angle_buffer) > buffer_size:
        angle_buffer.pop(0)
    return sum(angle_buffer) / len(angle_buffer)

def draw_gesture_progress(frame, angle_diff, threshold):
    """Draw progress bar for gesture completion"""
    progress = min(angle_diff / threshold, 1.0)
    bar_length = 200
    bar_filled = int(bar_length * progress)
    
    # Draw progress bar background
    cv2.rectangle(frame, (10, 200), (10 + bar_length, 220), (50, 50, 50), -1)
    # Draw progress bar fill
    color = (0, 255, 0) if progress >= 1.0 else (0, 255, 255)
    cv2.rectangle(frame, (10, 200), (10 + bar_filled, 220), color, -1)
    
    # Add text
    cv2.putText(frame, f"Progress: {progress*100:.0f}%", 
               (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_confidence_indicator(frame, confidence_level):
    """Draw confidence indicator circle"""
    color = (0, 255, 0) if confidence_level > 0.8 else (0, 255, 255) if confidence_level > 0.6 else (0, 0, 255)
    cv2.circle(frame, (frame.shape[1] - 30, 30), 15, color, -1)
    cv2.putText(frame, f"{confidence_level:.1f}", (frame.shape[1] - 40, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def get_finger_points(landmarks):
    """Get key finger points"""
    try:
        # Index finger
        index_mcp = landmarks[5]
        index_tip = landmarks[8]
        # Middle finger
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        # Ring finger
        ring_mcp = landmarks[13]
        ring_tip = landmarks[16]
        # Pinky finger
        pinky_mcp = landmarks[17]
        pinky_tip = landmarks[20]
        # Palm center (use middle of hand)
        palm_center = landmarks[9]  # Using middle finger MCP as palm reference
        
        return index_mcp, index_tip, middle_mcp, middle_tip, ring_mcp, ring_tip, pinky_mcp, pinky_tip, palm_center
    except (IndexError, AttributeError):
        return [None] * 9

def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    """Check if a specific finger is extended"""
    try:
        return landmarks[tip_idx].y < landmarks[pip_idx].y < landmarks[mcp_idx].y
    except (IndexError, AttributeError):
        return False

def is_index_finger_extended(landmarks):
    """Check if index finger is extended"""
    return is_finger_extended(landmarks, 8, 6, 5)

def is_middle_finger_extended(landmarks):
    """Check if middle finger is extended"""
    return is_finger_extended(landmarks, 12, 10, 9)

def is_ring_finger_extended(landmarks):
    """Check if ring finger is extended"""
    return is_finger_extended(landmarks, 16, 14, 13)

def is_pinky_finger_extended(landmarks):
    """Check if pinky finger is extended"""
    return is_finger_extended(landmarks, 20, 18, 17)

def calculate_distance_to_palm(fingertip, palm_center):
    """Calculate distance from fingertip to palm center"""
    try:
        dx = fingertip.x - palm_center.x
        dy = fingertip.y - palm_center.y
        return math.sqrt(dx*dx + dy*dy)
    except (AttributeError, ValueError):
        return 0

def calculate_angle_from_horizontal(vector):
    """Calculate angle of vector from horizontal"""
    try:
        angle_rad = math.atan2(vector[1], vector[0])
        angle_deg = math.degrees(angle_rad)
        return angle_deg + 360 if angle_deg < 0 else angle_deg
    except (ValueError, TypeError):
        return 0

def get_angle_difference(angle1, angle2):
    """Calculate the smallest angle difference between two angles"""
    try:
        diff = abs(angle1 - angle2)
        return diff if diff <= 180 else 360 - diff
    except (ValueError, TypeError):
        return 0

def normalize_vector(vector):
    """Normalize a vector to unit length"""
    try:
        magnitude = np.linalg.norm(vector)
        return vector / magnitude if magnitude > 0 else np.array([0, 0])
    except (ValueError, TypeError):
        return np.array([0, 0])

def is_gesture_locked(current_time, last_action_time, lock_duration):
    """Check if gesture is currently locked"""
    return current_time - last_action_time < lock_duration

def reset_tracking():
    """Reset all tracking variables"""
    global tracking_active, initial_finger_vectors, current_gesture_type, angle_buffer
    tracking_active = False
    initial_finger_vectors.clear()
    current_gesture_type = None
    angle_buffer.clear()

print("=== Enhanced YouTube Shorts Gesture Control ===")
print("GESTURES:")
print(f"1. INDEX FINGER UP (>{index_angle_threshold}°) → DOWN arrow (next video) + 2sec lock")
print("2. OPEN HAND → CLOSE TO FIST → UP arrow (previous video) + 2sec lock")
print("3. CONTROLS:")
print("   - Press 'q' to quit")
print("   - Press 'r' to reset tracking")
print("   - Press 's' to decrease sensitivity (increase threshold)")
print("   - Press 'a' to increase sensitivity (decrease threshold)")
print("   - Press 'h' to show/hide help")

show_help = True

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        
        frame_count += 1
        
        # More efficient memory management
        if frame_count % 50 == 0:
            gc.collect()
        
        # Reset accumulated data periodically
        if frame_count % 200 == 0:
            if len(angle_buffer) > buffer_size * 2:
                angle_buffer = angle_buffer[-buffer_size:]
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Adaptive frame processing
        should_process = (frame_count % process_every_n_frames == 0)
        
        if should_process:
            # Process the frame
            results = hands.process(rgb_frame)
        else:
            # Use previous results if available
            pass
        
        current_time = time.time()
        
        # Check if gesture is locked
        gesture_locked = is_gesture_locked(current_time, last_action_time, gesture_lock_duration)
        
        # Enhanced UI - Draw instructions and status
        if show_help:
            cv2.putText(frame, "INDEX UP = Next | OPEN HAND→FIST = Previous", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Sensitivity: {index_angle_threshold}° | 'a'=more 's'=less", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Current threshold display
        cv2.putText(frame, f"Threshold: {index_angle_threshold}°", 
                   (frame.shape[1] - 150, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        if gesture_locked:
            remaining_lock = gesture_lock_duration - (current_time - last_action_time)
            cv2.putText(frame, f"GESTURE LOCKED: {remaining_lock:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Adaptive processing based on hand detection
        if 'results' in locals() and results.multi_hand_landmarks and not gesture_locked:
            no_hands_counter = 0
            process_every_n_frames = 1
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # Get detection confidence
                    confidence = 0.8  # Default confidence
                    
                    # Draw confidence indicator
                    draw_confidence_indicator(frame, confidence)
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get hand landmarks
                    landmarks = hand_landmarks.landmark
                    
                    # Get finger points
                    finger_points = get_finger_points(landmarks)
                    if None in finger_points:
                        continue
                    
                    index_mcp, index_tip, middle_mcp, middle_tip, ring_mcp, ring_tip, pinky_mcp, pinky_tip, palm_center = finger_points
                    
                    # Convert to pixel coordinates
                    h, w, _ = frame.shape
                    index_mcp_px = (int(index_mcp.x * w), int(index_mcp.y * h))
                    index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))
                    middle_tip_px = (int(middle_tip.x * w), int(middle_tip.y * h))
                    ring_tip_px = (int(ring_tip.x * w), int(ring_tip.y * h))
                    pinky_tip_px = (int(pinky_tip.x * w), int(pinky_tip.y * h))
                    palm_center_px = (int(palm_center.x * w), int(palm_center.y * h))
                    
                    # Check finger states
                    index_extended = is_index_finger_extended(landmarks)
                    middle_extended = is_middle_finger_extended(landmarks)
                    ring_extended = is_ring_finger_extended(landmarks)
                    pinky_extended = is_pinky_finger_extended(landmarks)
                    
                    # Count extended fingers
                    extended_fingers = sum([index_extended, middle_extended, ring_extended, pinky_extended])
                    
                    # Display finger count for debugging
                    cv2.putText(frame, f"Extended fingers: {extended_fingers}", 
                               (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Determine current gesture type
                    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                        gesture_type = "index_only"
                    elif extended_fingers >= 2:  # Made even more lenient - 2+ fingers
                        gesture_type = "open_hand"
                    else:
                        gesture_type = "closed_hand"
                    
                    # Special case: if tracking open hand and fingers close, trigger action
                    if (current_gesture_type == "open_hand" and gesture_type == "closed_hand" and 
                        tracking_active and extended_fingers <= 1):
                        try:
                            pyautogui.press('up')
                            cv2.putText(frame, "HAND CLOSED! (UP ARROW)", (10, 180), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)
                            print(f"Hand closed! Extended fingers: {extended_fingers} - Pressing UP arrow")
                            
                            last_action_time = current_time
                            reset_tracking()
                            current_gesture_type = gesture_type
                            continue
                        except Exception as e:
                            print(f"Error pressing key: {e}")
                    
                    # Reset tracking if gesture type changes (except for the special case above)
                    if current_gesture_type is not None and current_gesture_type != gesture_type:
                        reset_tracking()
                        print(f"Gesture type changed from {current_gesture_type} to {gesture_type}")
                    
                    current_gesture_type = gesture_type
                    
                    # Enhanced finger visualizations
                    finger_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Blue, Red, Yellow
                    finger_positions = [index_tip_px, middle_tip_px, ring_tip_px, pinky_tip_px]
                    finger_states = [index_extended, middle_extended, ring_extended, pinky_extended]
                    
                    for i, (tip_px, extended, color) in enumerate(zip(finger_positions, finger_states, finger_colors)):
                        if extended:
                            cv2.circle(frame, tip_px, 8, color, -1)
                            # Add glow effect
                            cv2.circle(frame, tip_px, 12, color, 2)
                        else:
                            cv2.circle(frame, tip_px, 4, color, 2)  # Hollow circle for closed fingers
                    
                    # Enhanced palm center visualization
                    cv2.circle(frame, palm_center_px, 10, (255, 255, 255), 2)
                    cv2.circle(frame, palm_center_px, 5, (255, 255, 255), -1)
                    
                    hand_id = f"hand_{hand_idx}"
                    
                    # GESTURE 1: Index finger only extended (for upward movement = next video)
                    if gesture_type == "index_only":
                        cv2.putText(frame, "INDEX FINGER DETECTED", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Calculate index finger vector
                        index_vector = np.array([index_tip_px[0] - index_mcp_px[0], 
                                               index_tip_px[1] - index_mcp_px[1]])
                        index_vector = normalize_vector(index_vector)
                        
                        # Initialize tracking if not started
                        if not tracking_active:
                            initial_finger_vectors[hand_id] = {'index': index_vector.copy()}
                            tracking_active = True
                            cv2.putText(frame, "INDEX TRACKING STARTED", (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Calculate angle change for index finger
                        if tracking_active and hand_id in initial_finger_vectors:
                            if 'index' in initial_finger_vectors[hand_id]:
                                initial_index = initial_finger_vectors[hand_id]['index']
                                
                                # Calculate angles
                                initial_angle = calculate_angle_from_horizontal(initial_index)
                                current_angle = calculate_angle_from_horizontal(index_vector)
                                
                                # Calculate angle difference
                                raw_angle_diff = get_angle_difference(initial_angle, current_angle)
                                
                                # Apply smoothing
                                angle_diff = smooth_angle(raw_angle_diff)
                                
                                # Check if moved upward
                                movement_up = index_vector[1] < initial_index[1] - 0.1
                                
                                # Display information
                                cv2.putText(frame, f"Index Angle: {angle_diff:.1f}°", 
                                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Draw progress bar
                                draw_gesture_progress(frame, angle_diff, index_angle_threshold)
                                
                                # Check for upward movement trigger (13 degree threshold)
                                if angle_diff > index_angle_threshold and movement_up:
                                    try:
                                        pyautogui.press('down')
                                        cv2.putText(frame, "NEXT VIDEO! (DOWN ARROW)", (10, 180), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                                        print(f"Index UP movement! Angle: {angle_diff:.1f}° - Pressing DOWN arrow")
                                        
                                        last_action_time = current_time
                                        reset_tracking()
                                    except Exception as e:
                                        print(f"Error pressing key: {e}")
                            else:
                                reset_tracking()
                    
                    # GESTURE 2: Open hand (for convergence detection)
                    elif gesture_type == "open_hand":
                        cv2.putText(frame, "OPEN HAND DETECTED - CLOSE TO TRIGGER", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        
                        # Calculate distances from fingertips to palm center
                        distances = []
                        fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
                        
                        for fingertip in fingertips:
                            distance = calculate_distance_to_palm(fingertip, palm_center)
                            distances.append(distance)
                        
                        # Calculate average distance
                        avg_distance = sum(distances) / len(distances)
                        
                        # Initialize tracking if not started
                        if not tracking_active:
                            initial_finger_vectors[hand_id] = {'initial_distance': avg_distance}
                            tracking_active = True
                            cv2.putText(frame, "OPEN HAND TRACKING STARTED", (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            print(f"Open hand tracking started. Initial distance: {avg_distance:.3f}")
                        
                        # Display current distance for debugging
                        if tracking_active and hand_id in initial_finger_vectors:
                            if 'initial_distance' in initial_finger_vectors[hand_id]:
                                initial_distance = initial_finger_vectors[hand_id]['initial_distance']
                                convergence = initial_distance - avg_distance
                                
                                cv2.putText(frame, f"Convergence: {convergence:.3f} (need: {convergence_threshold:.3f})", 
                                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    elif gesture_type == "closed_hand":
                        cv2.putText(frame, "CLOSED HAND DETECTED", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    else:
                        # Reset tracking if no valid gesture
                        if tracking_active:
                            reset_tracking()
                            
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")
                    continue
        
        elif 'results' in locals() and (not results.multi_hand_landmarks or not hasattr(results, 'multi_hand_landmarks')) and not gesture_locked:
            # No hands detected - adaptive processing
            no_hands_counter += 1
            if no_hands_counter > 30:  # After 30 frames of no hands
                process_every_n_frames = 3  # Process every 3rd frame
            
            cv2.putText(frame, "NO HANDS DETECTED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            reset_tracking()
        
        # Enhanced status display
        if not gesture_locked:
            status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
            status_text = f"TRACKING: {current_gesture_type or 'NONE'}" if tracking_active else "TRACKING INACTIVE"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show frame
        cv2.imshow('Enhanced YouTube Shorts Gesture Control', frame)
        
        # Handle key presses with enhanced controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracking and clear gesture lock
            reset_tracking()
            last_action_time = 0
            print("Tracking and gesture lock reset!")
        elif key == ord('s'):  # Decrease sensitivity (increase threshold)
            index_angle_threshold = min(index_angle_threshold + 1, 25)
            print(f"Sensitivity decreased. Threshold increased to {index_angle_threshold}°")
        elif key == ord('a'):  # Increase sensitivity (decrease threshold)
            index_angle_threshold = max(index_angle_threshold - 1, 5)
            print(f"Sensitivity increased. Threshold decreased to {index_angle_threshold}°")
        elif key == ord('h'):  # Toggle help display
            show_help = not show_help
            print(f"Help display {'enabled' if show_help else 'disabled'}")

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Cleanup
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Enhanced gesture control stopped.")
