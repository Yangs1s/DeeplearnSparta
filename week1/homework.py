import cv2

cap = cv2.VideoCapture('/Users/yangseongjin/Desktop/DeepLearningSparta/week1/03.mp4')

while True:
    ret, img = cap.read()

    if not ret:
        break
    
    
    cropped_img = img[183:465, 721:878]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    # cropped_img = img[721:878, 183:465]
    # img = img[100:500, 150:650]
    cv2.imshow('cropped_img', cropped_img)
    cv2.imshow('result', img)

    if cv2.waitKey(1) == ord('q'):
        break