from flask import Flask, render_template,url_for, redirect, request, jsonify
import os
import numpy as np

from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.models import load_model


app = Flask(__name__)

model = load_model("lungs_cancer.h5")
print("Loaded model from disk")
class_label = ["Lungs Cancer Detected", "Normal"]

def load_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor = img_tensor/255
    return img_tensor

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        f = request.files['file']

        save_path = 'uploads/lungs'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, 'uploaded_image.jpg')

        f.save(file_path)
        
        # Make prediction

        loaded_image = load_image(f)

        prediction = model.predict(loaded_image)
        class_id = np.argmax(prediction, axis=1)
        output=class_label[int(class_id)]
        result = str(output)
        
        return result
    return None


if __name__ == "__main__":
    app.run(debug=True)
