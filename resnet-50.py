from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import numpy as np
from PIL import Image
import requests
from torchvision import transforms

import struct

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# print(model)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("./000000039769.jpg")
# image = Image.open("./13.jpg")
transform = transforms.Compose([transforms.ToTensor()])
img_convert_to_numpy = np.array(image)
img_convert_to_tensor2 = transform(img_convert_to_numpy)  # torch.Size([3, 480, 640])

# print(img_convert_to_numpy.shape)
# print(img_convert_to_tensor2)

# img_bytes = img_convert_to_tensor2.detach().numpy().astype("float32").tobytes()
# with open("image.bin", "wb") as file:
#     file.write(struct.pack("4i", *[2, 3, 480, 640]))
#     file.write(img_bytes)
#     file.write(img_bytes)


# model["classifier.1.weight"]
# outputs = model(torch.unsqueeze(img_convert_to_tensor2, 0),)
# print(outputs.logits)
outputs = model(torch.unsqueeze(img_convert_to_tensor2, 0),)
print(outputs.logits.softmax(-1))

# predicted_label = outputs.logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])


outputs = model.resnet.embedder.embedder.convolution(torch.unsqueeze(img_convert_to_tensor2, 0)) # [2, 64, 240, 320]
# print(outputs)
outputs = model.resnet.embedder.embedder.normalization(outputs) # [2, 64, 240, 320]
# print(outputs)
outputs = model.resnet.embedder.embedder.activation(outputs) # [2, 64, 240, 320]
# print(outputs)
outputs = model.resnet.embedder.pooler(outputs) # [2, 64, 120, 160]
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[0].layers[0].shortcut.convolution(outputs) # [2, 256, 120, 160]
# # print(outputs)
outputs = model.resnet.encoder.stages[0].layers[0].shortcut.normalization(outputs) # [2, 256, 120, 160]
# print(outputs)
residual = model.resnet.encoder.stages[0].layers[0].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[0].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[0].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[0].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[0].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[0].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[0].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[0].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[0].layers[0].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[0].layers[1].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[0].layers[1].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[1].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[1].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[1].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[1].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[1].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[1].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[1].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[0].layers[1].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[0].layers[2].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[0].layers[2].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[2].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[2].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[2].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[2].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[2].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[0].layers[2].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[0].layers[2].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[0].layers[2].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[1].layers[0].shortcut.convolution(outputs) # [2, 256, 120, 160]
# # print(outputs)
outputs = model.resnet.encoder.stages[1].layers[0].shortcut.normalization(outputs) # [2, 256, 120, 160]
# print(outputs)
residual = model.resnet.encoder.stages[1].layers[0].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[0].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[0].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[0].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[0].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[0].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[0].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[0].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[1].layers[0].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[1].layers[1].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[1].layers[1].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[1].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[1].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[1].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[1].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[1].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[1].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[1].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[1].layers[1].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[1].layers[2].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[1].layers[2].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[2].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[2].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[2].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[2].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[2].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[2].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[2].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[1].layers[2].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[1].layers[3].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[1].layers[3].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[3].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[3].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[3].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[3].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[3].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[1].layers[3].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[1].layers[3].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[1].layers[3].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[2].layers[0].shortcut.convolution(outputs) # [2, 256, 120, 160]
# # print(outputs)
outputs = model.resnet.encoder.stages[2].layers[0].shortcut.normalization(outputs) # [2, 256, 120, 160]
# print(outputs)
residual = model.resnet.encoder.stages[2].layers[0].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[0].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[0].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[0].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[0].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[0].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[0].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[0].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[2].layers[0].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[2].layers[1].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[2].layers[1].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[1].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[1].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[1].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[1].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[1].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[1].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[1].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[2].layers[1].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[2].layers[2].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[2].layers[2].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[2].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[2].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[2].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[2].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[2].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[2].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[2].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[2].layers[2].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[2].layers[3].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[2].layers[3].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[3].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[3].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[3].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[3].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[3].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[3].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[3].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[2].layers[3].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[2].layers[4].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[2].layers[4].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[4].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[4].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[4].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[4].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[4].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[4].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[4].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[2].layers[4].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[2].layers[5].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[2].layers[5].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[5].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[5].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[5].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[5].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[5].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[2].layers[5].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[2].layers[5].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[2].layers[5].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[3].layers[0].shortcut.convolution(outputs) # [2, 256, 120, 160]
# # print(outputs)
outputs = model.resnet.encoder.stages[3].layers[0].shortcut.normalization(outputs) # [2, 256, 120, 160]
# print(outputs)
residual = model.resnet.encoder.stages[3].layers[0].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[0].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[0].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[0].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[0].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[0].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[0].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[0].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[3].layers[0].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[3].layers[1].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[3].layers[1].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[1].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[1].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[1].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[1].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[1].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[1].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[1].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[3].layers[1].activation(outputs)
# print(outputs)
residual = outputs
outputs = model.resnet.encoder.stages[3].layers[2].shortcut(outputs)
# print(outputs)
residual = model.resnet.encoder.stages[3].layers[2].layer[0].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[2].layer[0].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[2].layer[0].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[2].layer[1].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[2].layer[1].normalization(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[2].layer[1].activation(residual) # [2, 256, 120, 160]
# print(residual)
residual = model.resnet.encoder.stages[3].layers[2].layer[2].convolution(residual) # [2, 64, 120, 160]
# print(residual.shape)
residual = model.resnet.encoder.stages[3].layers[2].layer[2].normalization(residual) # [2, 256, 120, 160]
# print(residual)
outputs = residual + outputs
outputs = model.resnet.encoder.stages[3].layers[2].activation(outputs)
# print(outputs)
outputs = model.resnet.pooler(outputs)
# print(outputs)
outputs = torch.flatten(outputs, start_dim=1)
# print(outputs)
outputs = model.classifier(outputs)
# print(outputs)
# print(model.classifier[1].weight.shape)
# print(model.classifier[1].bias.shape)
# outputs.fl
# print(outputs[0].shape)
# print(model.resnet.embedder.embedder.convolution.weight) # [64, 3, 7, 7]
# print(model.resnet.encoder.stages[0].layers[0].shortcut.convolution.weight) # [256, 64, 1, 1]
# print(model.resnet.encoder.stages[0].layers[0].shortcut.normalization.weight) # [256]
# print(model.resnet.encoder.stages[0].layers[0].shortcut.normalization.bias) # [256]
# print(model.resnet.encoder.stages[0].layers[0].shortcut.normalization.running_var.shape) # [256]
# print(model.resnet.encoder.stages[0].layers[0].layer[0].convolution.weight) # [64, 64, 1, 1]
# print(model.resnet.encoder.stages[0].layers[0].layer[0].convolution.weight)

# print(model.resnet.encoder.stages[1].layers[0].layer[0].convolution.weight) 
# print(model.resnet.encoder.stages[0].layers[2].layer[2].normalization.running_var.shape)
# print(model.resnet.encoder.stages[1].layers[0].shortcut.convolution.weight)
# print(model.resnet.encoder.stages[1].layers[0].layer[0].convolution.weight) 
# print(model.resnet.encoder.stages[0].layers[1].layer[0].convolution.weight) # [64, 64, 1, 1]