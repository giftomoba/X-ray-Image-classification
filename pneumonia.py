import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st


model = load_model('model_vgg3.hdf5')

st.title('X-ray Classification App')
st.write('For predicting Covid, Viral Pneumonia and Normal conditions')

uploaded_file = st.file_uploader('Upload an X-ray image...', type=['jpg', 'jpeg', 'png', 'gif'])

# Check if image is uploader:
if uploaded_file is not None:
    # Display the image:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    image = np.array(image)
    st.write('')
    
    if len(image.shape) < 3:
        image1 = np.expand_dims(image, axis=-1)
        #image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        res_image = np.resize(image1, (128,128))
        img = np.array(res_image).reshape(-1, 128, 128, 3)
        
        # Rescale the image:
        img = img / 255.0

        # Make Pedictions:
        prediction = np.argmax(model.predict(img))
        if prediction == 0:
            st.write(f'## Predicted Image is: Covid')
        elif prediction == 1:
            st.write(f'## Predicted Image is: Normal')
        else:
            st.write(f'## Predicted Image is: Viral Pneumonia')
        #label = 'Brain Tumour' if prediction[0][0]>0.5 else 'No Tumour'
    else:
        #image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        res_image = np.resize(image, (128,128))
        img = np.array(res_image).reshape(-1, 128, 128, 3)
        
        # Rescale the image:
        img = img / 255.0

        # Make Pedictions:
        prediction = np.argmax(model.predict(img))
        if prediction == 0:
            st.write(f'## Predicted Image is: Covid')
        elif prediction == 1:
            st.write(f'## Predicted Image is: Normal')
        else:
            st.write(f'## Predicted Image is: Viral Pneumonia')
        #label = 'Brain Tumour' if prediction[0][0]>0.5 else 'No Tumour'
