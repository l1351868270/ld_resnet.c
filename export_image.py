from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import numpy as np
from PIL import Image
import requests
from torchvision import transforms

import struct

transform = transforms.Compose([transforms.ToTensor()])
img_convert_to_tensor0 = transform(np.array(Image.open("./0.jpg")))
img_convert_to_tensor1 = transform(np.array(Image.open("./1.jpg")))
img = Image.open("./1.jpg")
# img_bytes = img_convert_to_tensor2.detach().numpy().astype("float32").tobytes()
with open("image.bin", "wb") as file:
    file.write(struct.pack("4i", *[2, img_convert_to_tensor1.shape[0], img_convert_to_tensor1.shape[1], img_convert_to_tensor1.shape[2]]))
    file.write(img_convert_to_tensor1.detach().numpy().astype("float32").tobytes())
    file.write(img_convert_to_tensor1.detach().numpy().astype("float32").tobytes())