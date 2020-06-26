import cv2
import numpy as np

dolly_path = './data/dolly.npy'
data = np.load(dolly_path)

first = data[0].reshape((100,100,3))

cv2.imshow('dolly',first)
cv2.waitKey(0)
cv2.destroyAllWindows()