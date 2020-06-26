import cv2
import numpy as np

# init camera
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter name of the person : ")

while True:
	ret,frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	# we want largest face acc to area, therefore sort and start loop from end
	faces = sorted(faces,key = lambda f: f[2]*f[3])

	# Pick last face
	face_selection = 0
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# Extract (Crop out required face) : Region of interest
		offset = 10
		face_selection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_selection,(100,100))

		skip+=1

		# save every 10th face
		if skip%10==0:
			face_data.append(face_selection)
			print(len(face_data))


	cv2.imshow('Frame',frame)
	cv2.imshow('Face selection',face_selection)

	key_pressed = cv2.waitKey(1) & 0xFF

	if key_pressed == ord('q'):
		break

# convert face list to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape( face_data.shape[0],-1 )
print(face_data.shape)

# save data into file system
np.save( dataset_path + file_name + '.npy' , face_data)
print("Data successfully saved at ", dataset_path + file_name + '.npy' )

cap.release()
cv2.destroyAllWindows()