import cv2
from tracker import*

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway1.mp4")

#Object detection from stable camera

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

#This will basiclly extract the moving objects from the frame

while True:
    #extracting every frame from the video
    ret, frame = cap.read()

    #Extracting the important area from the video which the moving parts are on 

    rio = frame[75: 720, 200: 800]
    #rio = frame[340: 720, 500: 800]

    #Object dection
    #Applying the detection on the frame

    mask = object_detector.apply(frame)
    #Removing the values that are gray(shadows)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contouns, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contouns:
        #Calculating area and prevent small elements to be highlighted
        area = cv2.contourArea(cnt)
        if area > 100: 
            #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2 )
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # Object Tracking 
    boxes_ids = tracker.update(detections)

    for boxes_id in boxes_ids:
        x, y, w, h, id = boxes_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2 )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)


    key = cv2.waitKey(30)
    if key == 27:
        break
        #Breaking the loop when pressing the key esc button


cap.release()
cv2.destroyAllWindows()

