import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/models/instance_norm/starry_night.t7')

img = cv2.imread('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/imgs/hw.jpeg')
img2 = cv2.imread('/Users/yangseongjin/Desktop/DeepLearningSparta/week2/imgs/01.jpg')
cropped_img = img2[140:370, 480:810]

h, w, c =cropped_img.shape

cropped_img = cv2.resize(cropped_img, dsize=(500, int(h / w *500)))

print(img2.shape)

MEAN_VALUE = [103.939, 116.779, 123.680]
#추론 진행
blob = cv2.dnn.blobFromImage(img2, mean=MEAN_VALUE)

print(blob.shape)

net.setInput(blob)
output = net.forward()

output =output.squeeze().transpose((1,2,0))
output =output + MEAN_VALUE

output = np.clip(output, 0,255)
output = output.astype('uint8')


output = cv2.resize(output, (w, h))

img[140:370, 480:810] = output

cv2.imshow('output', output)
cv2.imshow('img2',img2)
cv2.imshow('img', img)
cv2.waitKey(0)