import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

# Get the service resource
sqs = boto3.resource('sqs')

# Create the queue. This returns an SQS.Queue instance
# queue = sqs.create_queue(QueueName='test2.fifo',  Attributes={'FifoQueue': "true"})

# Send data to queue
queue = sqs.get_queue_by_name(QueueName='test2.fifo')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=64, shuffle=True)

for batch_idx, (data, target) in enumerate(train_loader):
    
    response = queue.send_message(
        MessageBody =  "test",
        MessageAttributes = {
            "data" : {
                "DataType": "Binary",
                "BinaryValue": data.data.numpy().tostring()
            }
        },
        MessageGroupId = "datagroup",
        MessageDeduplicationId = "something"
    )
    break

# # Read Message from queue
# queue = sqs.get_queue_by_name(QueueName='test2.fifo')

# messages = queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=1, MessageAttributeNames=['data'])

# data = messages[0].message_attributes["data"]["BinaryValue"]
# data_np = np.fromstring(data).reshape(64, 28, 28, 1)
# t = torch.Tensor(data_np)


print(queue.url)