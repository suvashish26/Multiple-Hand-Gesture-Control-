import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # With Draw
    # hands = detector.findHands(img, draw=False) # No Draw
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        bbox1 = hand1["bbox"]  # Bounding Box info x, y, w, h
        centerPoint1 = hand1["center"]  # center of the hand cx, cy
        handType1 = hand1["type"]  # Hand Type Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2['lmList']  # Corrected key from 'lmlist' to 'lmList'
            bbox2 = hand2["bbox"]  # Bounding Box info x, y, w, h
            centerPoint2 = hand2["center"]  # center of the hand cx, cy
            handType2 = hand2["type"]  # Hand Type Left or Right
            fingers2 = detector.fingersUp(hand2)
            print(fingers1, fingers2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
