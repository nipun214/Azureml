import numpy as np
from tensorflow import keras
from keras.preprocessing import image
model = keras.models.load_model('my_model.h5')
path='cat.jpg'
img=image.load_img(path, target_size=(150, 150))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0:
    print(" is a dog")
else:
    print(" is a cat")