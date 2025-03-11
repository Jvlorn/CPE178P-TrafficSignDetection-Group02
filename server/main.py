from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect  # Import WebSocketDisconnect
from fastapi.responses import HTMLResponse
import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net 
import json
import base64

from PIL import Image
import numpy as np
import io

import mobilenet_ms as mn # see mobilenet_ms.py 

app = FastAPI()

# Load the Mindspore model
param_dict = load_checkpoint("ckpt/traffic_sign_model_steps_5-1_20.ckpt") #check if file is existing // changed to our traffic design model

# Create instances of the backbone and head
backbone = mn.MobileNetV2Backbone()
head = mn.MobileNetV2Head(input_channel=backbone.out_channels, num_classes=58) # 5 flower classes // changed to 58

num_class = 58  # class_name = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'} // changed to 58
net = mn.mobilenet_v2(num_class)

# Load model parameters.
ms.load_param_into_net(net, param_dict)
model = ms.Model(net)

# Preprocessing function // changed to fit our model
'''
def preprocess_image(image_bytes):
    # img = Image.open(io.BytesIO(image_bytes))
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...]
    return Tensor(img, ms.float32)
'''
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = img.transpose(2, 0, 1)  # Change to CHW format
    img = img[np.newaxis, ...]  # Add batch dimension
    return Tensor(img, ms.float32)
'''
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)
    output = net(input_data)
    predicted_class = np.argmax(output.asnumpy())
    predicted_class = int(predicted_class) 
    probs = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
    
    s_np = probs[0,:]
    probs_label = np.argmax(s_np)
    prob_score = probs[0,probs_label] * 100
    print(prob_score)

    return {"class": predicted_class, "score": prob_score}
'''
@app.post("/predict") # // changed to fit our model
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)
    output = net(input_data)
    predicted_class = np.argmax(output.asnumpy())
    predicted_class = int(predicted_class)
    probs = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
    prob_score = float(probs[0, predicted_class] * 100)
    return {"class": predicted_class, "score": prob_score}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            image_data = base64.b64decode(json.loads(data)["data"])  # Decode the image data
            input_data = preprocess_image(image_data)
            output = net(input_data)

            predicted_class = np.argmax(output.asnumpy())
            predicted_class_int = int(predicted_class)
            probs = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
            prob_score = float(probs[0, predicted_class_int] * 100)

            # Send the prediction back to the client
            await websocket.send_text(json.dumps({
                "type": "prediction",
                "class": predicted_class_int,
                "score": prob_score
            }))
    except WebSocketDisconnect:
        pass  # Handle disconnections gracefully
    '''
    try:
        while True:
            data = await websocket.receive_text() 
            class_name = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

            #--- Prediction logic ---
            image_data = base64.b64decode(json.loads(data)["data"])  # Decode the image data
            input_data = preprocess_image(image_data)
            output = net(input_data)

            predicted_class = np.argmax(output.asnumpy())
            predicted_class_int = int(predicted_class) 
            predicted_class_str = class_name[predicted_class_int]
            probs = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
            print(probs) #added to show score in the terminal
            s_np = probs[0,:]
            probs_label = np.argmax(s_np)
            prob_score = float(probs[0,probs_label] * 100)

            print("prob score: {}".format(prob_score))

            # ... send the prediction back to the client ...
            # await websocket.send_text(json.dumps({"type": "prediction", "class": predicted_class_str}))
            await websocket.send_text(json.dumps({"type": "prediction", "class": predicted_class_str, "score" : prob_score}))
    except WebSocketDisconnect:
        pass  # Handle disconnections gracefully
    '''

