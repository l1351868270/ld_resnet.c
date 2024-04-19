
import time
import argparse
import struct
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import ResNetForImageClassification


# class ResNetConvLayer(nn.Module):

class ResNetConfig:
    depths: list = [3, 4, 6, 3]
    embedding_size: int = 64
    hidden_sizes: list = [256, 512, 1024, 2048]
    num_channels: int = 3

class ResNet(nn.Module):
    @classmethod
    def from_pretrained(cls, model_type):
        print("loading weights from pretrained resnet: %s" % model_type)

        model_hf = ResNetForImageClassification.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # print(sd_hf.keys())
        return model_hf

def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        file.write(struct.pack("i", 20240416)) # magic
        file.write(struct.pack("4i", *config.depths)) # depths
        file.write(struct.pack("i", config.embedding_size)) # embedding_size
        file.write(struct.pack("4i", *config.hidden_sizes)) # depths
        sd = model.state_dict()
        # embedder 6
        write_fp32(sd["resnet.embedder.embedder.convolution.weight"], file) # [64, 3, 7, 7]
        # print(sd["resnet.embedder.embedder.convolution.weight"])
        write_fp32(sd["resnet.embedder.embedder.normalization.weight"], file) # [64]
        # print(sd["resnet.embedder.embedder.normalization.weight"].shape)
        write_fp32(sd["resnet.embedder.embedder.normalization.bias"], file) # [64]
        # print(sd["resnet.embedder.embedder.normalization.bias"].shape)
        write_fp32(sd["resnet.embedder.embedder.normalization.running_mean"], file) # [64]
        # print(sd["resnet.embedder.embedder.normalization.running_mean"].shape)
        write_fp32(sd["resnet.embedder.embedder.normalization.running_var"], file) # [64]
        # print(sd["resnet.embedder.embedder.normalization.running_var"].shape)
        # write_fp32(sd["resnet.embedder.embedder.normalization.num_batches_tracked"], file) # [] 375525
        # print(sd["resnet.embedder.embedder.normalization.num_batches_tracked"].shape)

        # # shortcut 4 * 6 = 24
        # for i in range(len(config.depths)):
        #     write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.convolution.weight"], file) # [256, 64, 1, 1]
        #     # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.convolution.weight"].shape)
        #     write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.weight"], file) # [256]
        #     # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.weight"].shape)
        #     write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.bias"], file) # [256]
        #     # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.bias"].shape)
        #     write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_mean"], file) # [256]
        #     # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_mean"].shape)
        #     write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_var"], file) # [64]
        #     # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_var"].shape)

        #     # write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.num_batches_tracked"], file) # []
        #     # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.num_batches_tracked"].shape)

        # (3 + 4 + 6 + 3) * 18 = 288
        for i in range(len(config.depths)):
            write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.convolution.weight"], file) # [256, 64, 1, 1]
            # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.convolution.weight"].shape)
            write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.weight"], file) # [256]
            # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.weight"].shape)
            write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.bias"], file) # [256]
            # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.bias"].shape)
            write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_mean"], file) # [256]
            # print(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_mean"].shape)
            write_fp32(sd[f"resnet.encoder.stages.{i}.layers.0.shortcut.normalization.running_var"], file) # [64]
            for j in range(config.depths[i]):
                for k in range(3):
                    write_fp32(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.convolution.weight"], file)
                    # print(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.convolution.weight"].shape)
                # for k in range(3):
                    write_fp32(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.weight"], file)
                    # print(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.weight"].shape)
                # for k in range(3):
                    write_fp32(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.bias"], file)
                    # print(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.bias"].shape)
                # for k in range(3):
                    write_fp32(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.running_mean"], file)
                    # print(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.running_mean"].shape)
                # for k in range(3):
                    write_fp32(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.running_var"], file)
                    # print(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.running_var"].shape)
                # for k in range(3):
                    # write_fp32(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.num_batches_tracked"], file)
                    # print(sd[f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}.normalization.num_batches_tracked"].shape)
        # 2
        write_fp32(sd["classifier.1.weight"], file) # [1000, 2048]
        # print(sd["classifier.1.weight"].shape)
        write_fp32(sd["classifier.1.bias"], file) # [1000]
        # print(sd["classifier.1.bias"].shape)
        # 6 + 24 + 288 + 2 = 320

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    device = "cpu"
    # load the GPT-2 model weights
    model = ResNet.from_pretrained("microsoft/resnet-50")
    config = model.config
    with open("id2label.bin", "wb") as file:
        id2label = config.id2label
        id2label = OrderedDict(id2label)
        
        # print(len(id2label[7]), id2label[7])
        for k, v in id2label.items():
            fmt = f"{len(v)}s"
            # print(fmt)
            bk = struct.pack("i", len(v))
            bv = struct.pack(f"{len(v)}s", bytes(v, 'utf-8'))
            file.write(bk)
            file.write(bv)
    filename = "microsoft-resnet-50.bin"
    write_model(model, filename)
