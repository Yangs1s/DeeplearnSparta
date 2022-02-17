import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/models/instance_norm/udnie.t7')
# 모델을 토치로부터 읽는다. 그리고 넷에 저장해주세요.
# 마치 커피의 커피머신이라고 생각하면된다.
img = cv2.imread('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/imgs/02.jpeg')
#커피원두라고 생각하면 된다.
h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))
#너무 이미지가 크면 결과가 안나올수도 있어서 리사이즈를 해주는것이 좋다. 가로 500 세로는 가로에 비율에 맞게 유지

img = img[152:513, 185:428]

print(img.shape)

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)


print(blob.shape)
#-------------------전처리끝--------------------------------------------
net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0)) #차원을 다시 줄임.
output += MEAN_VALUE #다시 MEAN_VALUE를 더한다.(전처리에선 뺴고)

output = np.clip(output, 0, 255)
output = output.astype('uint8')

cv2.imshow('output',output)
cv2.imshow('img',img)
cv2.waitKey(0)