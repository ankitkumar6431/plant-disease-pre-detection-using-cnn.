# utils/predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('model/plant_model.h5')
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def get_prediction(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    label = labels[np.argmax(predictions)]
    return label
