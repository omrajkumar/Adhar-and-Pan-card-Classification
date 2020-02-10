from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pytesseract  
import  numpy as np
import cv2   
import os
import re

print('*****************************************')
print('        -:Output:-           ')
image_data = 'raj.jpg'
data = str(image_data)
# my_image = plt.imread("raj.jpg")
# plt.imshow(my_image)

# dimensions of our images
img_width, img_height = 64, 64

# load the model we saved
model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
img = image.load_img(data, target_size=(img_width, img_height))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
if classes[0][0] == 1:
    pred = 'PAN CARD'
else:
    pred = 'ADHAR CARD'
print ('You Have Given...! ')
print (pred)

#####################################
try:
    image = data
#     print('Editing image for better OCR result..........')
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    new_image = 'edited' + '_' + image 
    cv2.imwrite(new_image, img)
    read = pytesseract.image_to_string(new_image)
#     print(read)

except Exception as e:
    print('please provide proper name of the image')
    print(e)
####################################

if classes[0][0] == 1:
    numbers = re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]{1}',read)
else:
    numbers = re.findall(r'[0-9][0-9 .\-\(\)]{4,}', read)

#print(numbers)

for number in numbers:
    print(pred +' No :-> ' + number)
    F = open('output.json', 'a+')
    F.write(pred +'\n:-> ' + number)    
print('*****************************************')

