from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import boto3
import pickle

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

model1 = Section1()

# Create s3 client
# Weird - us-east-1 is default so don't specify it when you are using it?
region = "us-east-1"
bucket_name = "serverless-decouplednn-49329"
s3_client = boto3.client('s3', region_name = region)

existing_buckets = s3_client.list_buckets()

# Output the bucket names
print('Existing buckets:')
for bucket in existing_buckets['Buckets']:
    print(f'  {bucket["Name"]}')

if bucket_name not in existing_buckets['Buckets']:
    # Create unique name
    s3_client.create_bucket(Bucket=bucket_name)

# upload
# save_data = model1.state_dict()
# save_data = pickle.dumps(save_data)

# # unique layer name
layer_name = "layer_1.pkl"
# s3 = boto3.resource('s3')
# obj = s3.Object(bucket_name, layer_name)
# obj.put(Body=save_data)

#response = s3_client.put_item()


s3 = boto3.resource('s3')
obj = s3.Object(bucket_name, layer_name)
save_data = obj.get()["Body"].read()
save_data = pickle.loads(save_data)

model1.load_state_dict(save_data)

print("test")