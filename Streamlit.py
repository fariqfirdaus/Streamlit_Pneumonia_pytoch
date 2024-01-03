import streamlit as st
import torch
from PIL import Image

def load_model(file_path, map_location='cpu'):
    model = torch.load(file_path, map_location=map_location)
    return model
#def load_model():
#    model = PneumoniaResnet()
#    model.load_state_dict(torch.load('PneumoniaResnet.pth', map_location = torch.device('cpu')))
#    model.eval()
#    return model
    

model = load_model()

def predict(image):
    tensor_image = your_image_preprocessing_function(image)

    with torch.no_grad():
        output = model(tensor_image)
        prediction = process_model_output(output)
        return prediction
    
st.title('Pneumonia')
upload_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uplaod_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict(image)
        st.write(prediction)