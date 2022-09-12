# steps for execution
# in the below code you can see a variable named "video = cv2.VideoCapture('videos\Video2.mp4')"
# in that "video" variable give path of an video and then execute the code



import cv2

# get face.xml file data
tarinedDataSet = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# start the webcam
video = cv2.VideoCapture('videos\Video2.mp4')

while True:
    # --> read returns two values i.e succuss(true.false) and image
    success, frame = video.read()
    # convert image into black and white or gray-scale because openCV will work only for black and white images
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detech faces || detectMultiScale will return cordinates of detected faces
    # cordinated will be like [[38 165 279 279]] ie [[x y height width]]
    faceCordinates = tarinedDataSet.detectMultiScale(grayImage)
    for i in range(len(faceCordinates)):
        x, y, w, h = faceCordinates[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# show image
    cv2.imshow("Detected Face", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
video.release()
