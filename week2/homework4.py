import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/models/instance_norm/mosaic.t7')
net2 = cv2.dnn.readNetFromTorch('//Users/yangseongjin/Desktop/DeepLearningSparta/week2/models/instance_norm/feathers.t7')
net3 = cv2.dnn.readNetFromTorch('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/models/instance_norm/candy.t7')

#영상#########
cap = cv2.VideoCapture(0)

######## 이미지처리#########################################3
# img = cv2.imread('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/imgs/03.jpg')


while True:
    ret, img =cap.read()
    
    if ret == False:
        break

    h, w, c = img.shape

    img = cv2.resize(img, dsize=(500, int(h / w * 500)))

    MEAN_VALUE = [103.939, 116.779, 123.680]
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

    net.setInput(blob)
    output = net.forward()

    output = output.squeeze().transpose((1, 2, 0))

    output += MEAN_VALUE
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')


    net2.setInput(blob)
    output2 = net2.forward()

    output2 = output2.squeeze().transpose((1, 2, 0))

    output2 += MEAN_VALUE
    output2 = np.clip(output2, 0, 255)
    output2 = output2.astype('uint8')


    net2.setInput(blob)
    output2 = net2.forward()

    output2 = output2.squeeze().transpose((1, 2, 0))

    output2 += MEAN_VALUE
    output2 = np.clip(output2, 0, 255)
    output2 = output2.astype('uint8')

    net3.setInput(blob)
    output3 = net3.forward()

    output3 = output3.squeeze().transpose((1, 2, 0))

    output3 += MEAN_VALUE
    output3 = np.clip(output3, 0, 255)
    output3 = output3.astype('uint8')


    #이미지 반으로 자르기
    output = output[:150, :]
    output2 = output2[150:350, :]
    output3 = output3[350:, :]
    #이미지 합치기(y방향으로 합쳐주세요)
    output4 = np.concatenate([output,output2,output3],axis=0)


    cv2.imshow('output4',output4)
    cv2.waitKey(0)