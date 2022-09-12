import cv2
from random import randrange as randRange

# get face.xml file data
tarinedDataSet = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# select image for detecting face
# img = cv2.imread("Project\Face_Detection\images\Group1.jpg", 0) -----> 0 will convert image into black and white

img = cv2.imread("images\Group2.jpg")
# cv2.imshow("Picture", img)
# cv2.waitKey()  # it will hold the execution till a key is pressed
# print(img)


# convert image into black and white or gray-scale because openCV will work only for black and white images
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detech faces || detectMultiScale will return cordinates of detected faces
# cordinated will be like [[38 165 279 279]] ie [[x y height width]]
faceCordinates = tarinedDataSet.detectMultiScale(grayImage)
print(faceCordinates)

for i in range(len(faceCordinates)):
    x, y, w, h = faceCordinates[i]
    cv2.rectangle(img, (x, y), (x+w, y+h), (randRange(0, 256),
                  randRange(0, 256), randRange(0, 256)), 2)

# show image
cv2.imshow("Detected Face", img)
cv2.waitKey()
