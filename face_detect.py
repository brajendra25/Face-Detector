import cv2
import sys
count=1;
# Get user supplied values
imagePath = "test.jpeg"
cascPath = "haarcascade_frontalface_default.xml"
imageDict={}
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
#print("Image : ",image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("color :",gray)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    
    imageDict[count]=x,y
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    count=count+1
cv2.imshow("Faces found", image)
print("Images Data: ",imageDict)
cv2.waitKey(0)
