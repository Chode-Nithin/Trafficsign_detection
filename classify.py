# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img

import numpy as np
import streamlit as st
from PIL import Image

# @st.cache(allow_output_mutation=True)
# def get_model():
#         model = load_model(r'E:/sem6/traffic/traffic_classifier.h5')
#         print('Model Loaded')
#         return model 
import streamlit as st
import tensorflow as tf

@st.cache_resource()
def get_model():
    model_path =r'E:/sem6/traffic/traffic_classifier.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model
model = get_model()

        
def predict1(image):
        loaded_model = get_model()
        #image = load_img(image)
        image = tf.keras.preprocessing.image.load_img(image)
        image = image.resize((30,30))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        # predict_x=loaded_model.predict(image)
        # pred=np.argmax(predict_x,axis=1)
        # image = image.resize((30, 30))
        # image = img_to_array(image)
        # image = image/255.0
        # image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size

        classes = loaded_model.predict(image)
        pred = np.argmax(classes, axis=1)

        return pred
        # loaded_model = get_model()
        # image = load_img(image, target_size=(32, 32), color_mode = "grayscale")
        # image = img_to_array(image)
        # image = image/255.0
        # image = np.reshape(image,[1,32,32,1])

        # classes = loaded_model.predict(image)
        # pred=np.argmax(classes,axis=1)

        # return pred

        # loaded_model = get_model()
        # image = load_img(image)
        # image = image.resize((30,30))
        # #image = img_to_array(image)
        # #image = image/255.0
        
        # image = np.array(image)
        # #image = np.reshape(image,[1,30,30,3])
        # classes = loaded_model.predict(image)
        # pred=np.argmax(classes,axis=1)
    

        # return pred
