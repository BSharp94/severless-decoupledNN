import numpy as np
import os
import pickle
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
from google.cloud import storage, pubsub

def get_model_data(bucket_name, model_name):
    storage_client = storage.Client()

    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(model_name)
        model_data = pickle.loads(bucket.get_blob(model_name).download_as_string())
        return model_data
    except:
        # TODO - Add Logging
        return None

def save_model(bucket_name, model_name, model, cred_path = None):
    storage_client = storage.Client()

    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(model_name)

        model_data = pickle.dumps(model.state_dict())
        blob.upload_from_string(model_data) 

        return True
    except:
        # TODO - Add Logging
        return False

def encode_signal(signal):
    return str(base64.b64encode(signal.numpy().tobytes()))[2:-1]

def decode_signal(encoded_signal, shape):
    signal = np.frombuffer(base64.b64decode(encoded_signal), dtype=np.float32)
    signal = signal.reshape(shape)
    return torch.Tensor(signal)

class Section1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 36, kernel_size = 2, padding=1, bias = False)
        self.conv2 = nn.Conv2d(36, 36, kernel_size = 2, bias = False)
        self.pooling1 = nn.AvgPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p = 0.25)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pooling1(out)
        out = self.dropout1(out)
        return out

def run_section_forward(request):
    request_json = request.get_json()

    # Get Input Signal and Environment Vars
    input_signal = decode_signal(request_json["input_signal"], (64, 1, 28, 28))
    bucket_name = os.environ.get("BUCKET_NAME")
    model_name = os.environ.get("MODEL_NAME")
    
    # Load Model - If it fails then save the model
    model = Section1()
    model_state = get_model_data(bucket_name, model_name)
    if model_state:
        model.load_state_dict(model_state)
    else:
        save_model(bucket_name, model_name, model)
    
    # Send Signal Forward
    output = model.forward(input_signal)

    # Send Output to PubSub
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path("cloundnetwork", "section2-input")

    future = publisher.publish(
        topic_path, data=base64.b64encode(output.data.numpy().data)  # data must be a bytestring.
    )

    # Make network request to next layer 
    encoded_signal = encode_signal(output.data)
    data = {
        "input_signal": encoded_signal,
        "output_signal": request_json["output_signal"], # TODO make into pubsub
    }

    requests.post("https://us-central1-cloundnetwork.cloudfunctions.net/section2", json=data)
    
def run_section_backwards(request):
    request_json = request.get_json()

    # Get Backprop Signal and Environment Vars
    backprop_signal = decode_signal(request_json["backprop_signal"], (64, 36, 14, 14))
    bucket_name = os.environ.get("BUCKET_NAME")
    model_name = os.environ.get("MODEL_NAME")

    # Get Input Signal
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path("cloundnetwork", "section1-back")
    response = subscriber.pull(subscription_path, max_messages=1)
    messages = response.received_messages
    input_signal = decode_signal(messages[0].message.data, (64, 1, 28, 28))

    # Load Model - If it fails then save the model
    model = Section1()
    model_state = get_model_data(bucket_name, model_name)
    if model_state:
        model.load_state_dict(model_state)
    else:
        save_model(bucket_name, model_name, model)
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    # Send Signal Forward
    #input_signal.requires_grad_(True)
    output = model.forward(input_signal)

    # Calculate Loss
    output.backward(backprop_signal)

    # Save Model
    save_model(bucket_name, model_name, model)




