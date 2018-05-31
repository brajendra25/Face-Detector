import cv2
import sys
import numpy as np

count = 0
cascPath = "haarcascade_frontalface_default.xml" #sys.argv[0]
faceCascade = cv2.CascadeClassifier(cascPath)
faceMovingCordinates = {}
webcam_capture = cv2.VideoCapture(0)

while (webcam_capture.isOpened()):  #Check Webcam is working or Not
    # Capture frame-by-frame
    ret, frame = webcam_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #print(w,h)
         #Storing Face moment coordinates
        faceMovingCordinates[count]=(x,y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,51), 2)
        count = count + 1
        sub_face = frame[y:y+h, x:x+w]
        FaceFileName = "Detected_Faces/face_" + str(y) + ".jpg"
        #print(FaceFileName)
        cv2.imwrite(FaceFileName, sub_face)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
#print(faceMovingCordinates)
webcam_capture.release()
cv2.destroyAllWindows()