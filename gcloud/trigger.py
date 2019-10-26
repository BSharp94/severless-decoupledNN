
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

epoch = 0

futures = dict()

def get_callback(f, id):
    def callback(f):
        try:
            futures.pop(id)
        except:
            print("No Future found")
    return callback

for batch_idx, (data, target) in enumerate(train_loader):
    
    #batch_uid = 1000 * epoch + batch_idx
    batch_uid = f'{epoch:03}' + f'{batch_idx:04}'
    batch_uid_encoded = bytes(batch_uid, "utf-8")

    # Get single batch of data
    data_batch = data
    data_target = target

    # convert to numpy
    numpy_data = data_batch.data.numpy()
    byte_data = numpy_data.tobytes()

    # Send Output to PubSub
    publisher = pubsub.PublisherClient.from_service_account_json('./gcloud/creds.json')
    
    message_data = base64.b64encode(data_batch.data.numpy().data)

    # Send to Section1 input
    topic_path = publisher.topic_path("cloundnetwork", "section1_input")
    future = publisher.publish(
        topic_path, data=batch_uid_encoded + message_data # data must be a bytestring.
    )
    futures["section1_input"] = future
    future.add_done_callback(get_callback(future, "section1_input"))

    # Send to Section1 input Delay
    topic_path = publisher.topic_path("cloundnetwork", "section1_input_delay")
    future = publisher.publish(
        topic_path, data=batch_uid_encoded + message_data # data must be a bytestring.
    )
    futures["section1_input_delay"] = future
    future.add_done_callback(get_callback(future, "section1_input_delay"))

    # Send to Label
    topic_path = publisher.topic_path("cloundnetwork", "labels")
    future = publisher.publish(
        topic_path, data= bytes(data_target)
    )
    futures["labels"] = future
    future.add_done_callback(get_callback(future, "labels"))

    while futures:
        print("Delay for futures on batch: ", batch_idx)
        time.sleep(1)

    time.sleep(1)