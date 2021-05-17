import mediapipe as mp
import cv2
import msvcrt


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)


cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
    
    while cap.isOpened():
        # how do I change the read rate?
        # I imagine once I start doing calculations to figure out if the hands are near the face
        # things will slow down.
        # no need to read in as fast as possible.
        ret, frame = cap.read()
        # turns frame into mirror
        # frame = cv2.flip(frame,1)
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # prints coords of face / hands?
        # print(results.face_landmarks)
        # print(results.left_hand_landmarks)
        # print(results.right_hand_landmarks)
        # how do I access all of the points for each landmark though?  (face border, fingers, etc.)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        head_draw_spec = mp_drawing.DrawingSpec(color=(100,0,0), thickness=1, circle_radius=1)
        left_draw_spec = mp_drawing.DrawingSpec(color=(0,100,0), thickness=1, circle_radius=1)
        right_draw_spec = mp_drawing.DrawingSpec(color=(0,0,100), thickness=1, circle_radius=1)
        
        # Face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 head_draw_spec)
        
        #Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 right_draw_spec)

        #Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 left_draw_spec)


        cv2.imshow('Video Feed', image)

        exit = cv2.waitKey(10) # press ESC to exit
        if exit == 27:
            break

cap.release()
cv2.destroyAllWindows()