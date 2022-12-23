import cv2
import mediapipe
import numpy
import autopy





def handlandmarks(colorImg):
    # Default values if no landmarks are tracked
    landmarkList = []
    # Object for processing the video input
    landmarkPositions = mainHand.process(colorImg)
    landmarkChecker = landmarkPositions.multi_hand_landmarks


    # Stores the out of the processing object (returns False on empty)
    if landmarkChecker:
        for hand in landmarkChecker:
            for index, landmark in enumerate(hand.landmark):


                draw.draw_landmarks(img, hand, initHand.HAND_CONNECTIONS)

                h, w, c = img.shape  # Height, width and channel on the image
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([index, centerX, centerY])


    return landmarkList


def fingersmovements(landmarks):
    fingerTips = []  # To store 4 sets of 1s or 0s
    tipIds = [4, 8, 12, 16, 20]  # Indexes for the tips of each finger

    # Check if thumb is up
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0] - 1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)

    # Check if fingers are up except the thumb
    for id in range(1, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][2]:
            # Checks to see if the tip of the finger is higher than the joint
            fingerTips.append(1)
        else:
            fingerTips.append(0)

    return fingerTips

cap = cv2.VideoCapture(0)
# Initializing mediapipe
initHand = mediapipe.solutions.hands
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
# Making Object to draw the connections between each finger index
draw = mediapipe.solutions.drawing_utils
# Outputs the height and width of the screen (1920 x 1080)
wScr, hScr = autopy.screen.size()
# Previous x and y location
pX, pY = 0, 0
# Current x and y location
cX, cY = 0, 0

while True:
    check, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmList = handlandmarks(imgRGB)
    # cv2.rectangle(img, (75, 75), (640 - 75, 480 - 75), (255, 0, 255), 2)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Gets index 8s x and y values (skips index value because it starts from 1)

        finger = fingersmovements(lmList)  # Calling the fingers function to check which fingers are up

        if finger[1] == 1 and finger[2] == 0:  # Checks to see if the pointing finger is up and thumb finger is down
            x3 = numpy.interp(x1, (75, 640 - 75),(0, wScr))
            y3 = numpy.interp(y1, (75, 480 - 75),(0, hScr))

            cX = pX + (x3 - pX) / 2
            cY = pY + (y3 - pY) / 2

            autopy.mouse.move(wScr - cX,cY)  # Function to move the mouse to the x3 and y3 values (wSrc inverts the direction)
            pX, pY = cX, cY  # Stores the current x and y location as previous x and y location for next loop

        if finger[1] == 1 and finger[0] == 1:  # Checks to see if the pointer finger is down and thumb finger is up
            autopy.mouse.click()  # Left click

        if finger[1] == 0 and finger[4] == 1:
            break


    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break