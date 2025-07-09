from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time

start = time.time()

np.set_printoptions(suppress=True)

# Load model and labels once
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt")]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image).astype(np.float32)
    normalized_image = (image_array / 127.5) - 1
    data = np.expand_dims(normalized_image, axis=0)  # Shape (1, 224, 224, 3)

    prediction = model.predict(data)


    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]

    return class_name, confidence


# Example usage:
print(predict("test_cat.jpg"))
print(f"Time: {time.time() - start}")