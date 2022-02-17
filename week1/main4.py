import cv2

cap = cv2.VideoCapture('/Users/yangseongjin/Desktop/DeepLearningSparta/week1/04.mp4')

while True:
    ret, img = cap.read()

    if ret == False:
        break

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = img[100:500, 150:650]
    cv2.imshow('result', img)

    if cv2.waitKey(100) == ord('q'):
        break