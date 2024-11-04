import cv2

image = cv2.imread('pic/Opencv_logo.png')
image = cv2.resize(image, (256, 256))

cv2.imwrite('pic/Opencv_logo_256.jpg', image)