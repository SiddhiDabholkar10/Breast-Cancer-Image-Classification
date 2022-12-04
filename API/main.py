from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import cv2

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model=tf.keras.models.load_model("../breast_cancer_classifier.h5")

class_names=["Benign","Malignant"]


@app.get("/hello")
async def hello():
    return "hey server started hii"


def read_file_as_img(data):
    #image=np.array(Image.open(BytesIO(data)).resize((256,256)))
    image = np.array(Image.open(BytesIO(data)).resize((256,256)))
    return image


@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    
    img=read_file_as_img(await file.read())
    # newsize = (256,256)
    # img.resize(newsize)
    
    img_batch=np.expand_dims(img,0)

    result=model.predict(img_batch)
    
    finalprediction=class_names[np.argmax(result[0])]
    confidence=np.max(result[0])

    return {
        'class':finalprediction,
        'confidence':float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)