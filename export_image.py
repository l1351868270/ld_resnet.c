from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import numpy as np
from PIL import Image
import requests
from torchvision import transforms

import struct

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
image = Image.open("./0.jpg")
# image = Image.open("./1.jpg")
inputs = processor(images=image, return_tensors="pt")
inputs = inputs["pixel_values"]

# img_bytes = img_convert_to_tensor2.detach().numpy().astype("float32").tobytes()
with open("image.bin", "wb") as file:
    file.write(struct.pack("4i", *[2, inputs.shape[1], inputs.shape[2], inputs.shape[3]]))
    file.write(inputs.detach().numpy().astype("float32").tobytes())
    file.write(inputs.detach().numpy().astype("float32").tobytes())