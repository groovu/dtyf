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
        height, width, _ = image.shape

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

        hand_landmarks = results.right_hand_landmarks
        h, w, c = frame.shape
        if hand_landmarks:
            print(type(hand_landmarks.landmark))
            #cv2.rectangle(frame, (0,0), (hand_landmarks.x, hand_landmarks.y), (0,222,0))
        # if hand_landmarks:
        #     for handLMs in hand_landmarks:
        #         x_max = 0
        #         y_max = 0
        #         x_min = w
        #         y_min = h
        #         for lm in handLMs.landmark:
        #             x, y = int(lm.x * w), int(lm.y * h)
        #             if x > x_max:
        #                 x_max = x
        #             if x < x_min:
        #                 x_min = x
        #             if y > y_max:
        #                 y_max = y
        #             if y < y_min:
        #                 y_min = y
        #         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #         mp_drawing.draw_landmarks(frame, handLMs, mp_holistic.HAND_CONNECTIONS)

        # x = results.left_hand_landmarks
        # cv2.rectangle(frame, (0,0), (100,100), (0,255,0), 2)
        # print(x)
        cv2.imshow('Video Feed', image)

        exit = cv2.waitKey(10) # press ESC to exit
        if exit == 27:
            break

cap.release()
cv2.destroyAllWindows()