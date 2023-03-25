import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def classify_image(image):
    # Convert the image to a Pillow image object
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    # Resize the image
    img = img.resize((160, 160))
    # Convert the image to a numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Expand dimensions to match input target shape
    img_array = np.expand_dims(img_array, axis=0)
    # Create the model object
    model = load_model("softmax_2units")
    # Make prediction
    pred = model.predict(x=img_array)
    # Get the predicted class
    pred_class = np.argmax(pred)

    if pred_class == 0:
        return "Cat"
    if pred_class == 1:
        return "Dog"
    
examples = ['cat.4.jpg', 'cat.64.jpg', 'dog.4.jpg', 'dog.45.jpg']
description = "Upload a picture. Click submit"

interface = gr.Interface(fn=classify_image,
                          inputs=gr.Image(shape=(200, 200)),
                          outputs=gr.Text(),
                          examples=examples,
                          description=description,
                          flagging_options=['Correct', 'Wrong'])


interface.launch(debug=True)