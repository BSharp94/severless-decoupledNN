
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from google.cloud import storage, pubsub
import base64
import time
import asyncio
import requests

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=True)

for batch_idx, (data, target) in enumerate(train_loader):
    
    print("batch: " + str(batch_idx))

    # Get single batch of data
    data_batch = data
    data_target = target

    # convert to numpy
    numpy_data = data_batch.data.numpy()
    byte_data = numpy_data.tobytes()



    # Send Output to PubSub
    publisher = pubsub.PublisherClient.from_service_account_json('./_spike_gcloud2/creds.json')
    topic_path = publisher.topic_path("cloundnetwork", "section1-input")

    future = publisher.publish(
        topic_path, data=base64.b64encode(data_batch.data.numpy().data)  # data must be a bytestring.
    )

    data = {
        "input_signal": str(base64.b64encode(byte_data))[2:-1],
        "output_signal": "".join([str(i) for i in list(data_target.numpy())]),
    }
    print("tick")
    #signals.append(run_request(data))
    requests.post("https://us-central1-cloundnetwork.cloudfunctions.net/section1", json=data)
#    requests.post("https://us-central1-cloundnetwork.cloudfunctions.net/section1", json=data)

    print("tock")
    time.sleep(1)