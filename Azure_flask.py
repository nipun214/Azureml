from flask import Flask,request, render_template
import numpy as np
from tensorflow import keras
from keras.preprocessing import image

app = Flask(__name__)
model = keras.models.load_model('my_model.h5')


def model_predict(img_path):
    model = keras.models.load_model('my_model.h5')
    img=image.load_img(img_path, target_size=(150, 150))
    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    if classes[0]>0:
        return 'is a dog'
    else:
        return 'is a cat'

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    f = request.files['fileToUpload']
    return model_predict(f)

if __name__ == '__main__':
    app.run(debug=True)
