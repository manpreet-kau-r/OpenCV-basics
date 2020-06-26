import cv2

img = cv2.imread('my pic.jpg')
gray = cv2.imread('my pic.jpg',cv2.IMREAD_GRAYSCALE)

cv2.imshow('My image',gray)

cv2.waitKey(2500)
cv2.destroyAllWindows()

