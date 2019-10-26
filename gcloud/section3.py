import numpy as np
import os
import pickle
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
from google.cloud import storage, pubsub
import time

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

class Section3(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7 * 7 * 64, 1028)
        self.dropout3 = nn.Dropout(p = 0.25)
        self.linear2 = nn.Linear(1028, 10)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout3(out)
        out = F.softmax(self.linear2(out), dim=1)

        return out


futures = dict()

def get_callback(f, id):
    def callback(f):
        try:
            futures.pop(id)
        except:
            print("No Future found")
    return callback

def run_section_full(event, context):

    input_message = base64.b64decode(event['data']).decode("utf-8")
    batch_uid = input_message[:7]
    batch_uid_encoded = bytes(batch_uid, "utf-8")
    
    print(batch_uid)

    # Load input Signal
    input_signal = decode_signal(input_message[7:], (64, 64 * 7 * 7))
    bucket_name = os.environ.get("BUCKET_NAME")
    model_name = os.environ.get("MODEL_NAME")
    
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path("cloundnetwork", "labels")
    response = subscriber.pull(subscription_path, max_messages=1)
    messages = response.received_messages
    label = torch.Tensor(list(messages[0].message.data)).type(torch.LongTensor)

    # Acknowledge message
    subscriber.acknowledge(subscription_path,[messages[0].ack_id])

    # Load Model
    model = Section3()
    model_state = get_model_data(bucket_name, model_name)
    if model_state:
        model.load_state_dict(model_state)
    else:
        save_model(bucket_name, model_name, model)
    optimizer3 = torch.optim.Adam(model.parameters(), lr = 5e-4)

    # Send Signal Forward
    input_signal.requires_grad_(True)
    output = model.forward(input_signal)

    # Get True Output Signal
    loss = F.cross_entropy(output, label)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == label).sum().item()
    acc = correct / label.size(0)
    print("Acc: ", acc) # TODO send to record endpoint

    # Send signal back
    loss.backward()
    optimizer3.step()
    
    save_model(bucket_name, model_name, model)
    
    # Make network request to next layer 
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path("cloundnetwork", "section2_backprop")

    data = base64.b64encode(input_signal.grad.data.numpy().data) 

    future = publisher.publish(
        topic_path, data= batch_uid_encoded + data # data must be a bytestring.
    )
    futures["section2_backprop"] = future
    future.add_done_callback(get_callback(future, "section2_backprop"))

    while futures:
        time.sleep(1)