import functools

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models
from PIL import Image

import finetune
#
# class DenseDistractor(torch.nn.Module):
#     def __init__(self, input_shape, output_shape, n_dense=2):
#         super().__init__()
#         input_size = functools.reduce(lambda a,b: a*b, input_shape)
#         output_size = functools.reduce(lambda a,b: a*b, output_shape)
#         self.input_size = input_size
#         self.output_size = output_size
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         print(input_shape)
#         print(output_shape)
#         sizes = [int(input_size + np.round((i * (output_size - input_size))) / n_dense) for i in range(n_dense + 1)]
#         print(sizes)
#         sequence = []
#         for i in range(n_dense):
#             sequence.append(torch.nn.Linear(sizes[i], sizes[i+1]))
#             sequence.append(torch.nn.ReLU)
#         self.dense_layers = torch.nn.Sequential(*sequence)
#
#     def forward(self, x):
#         x = x.view(self.input_size)
#         x = self.dense_layers(x)
#         return x.view(*self.output_shape)

class Distractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1)  # preserve dimensions


    def forward(self, x):
        x = self.cnn(x)
        x = functional.relu(x)
        x = functional.softsign(x)
        return x


def get_target_layers(classifier):
    # print(classifier)

    target_layers = []
    if type(classifier) == models.ResNet: # todo: more models
        target_layers = [classifier.layer4[-1]]
    return target_layers

def grad_cam_from_batch(classifier, batch):
    # todo: understand how target_layers argument to gradcam works
    target_layers = get_target_layers(classifier)
    inputs, labels = batch
    with GradCAM(model=classifier, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        targets = [ClassifierOutputTarget(l) for l in labels.tolist()]
        grayscale_cams = cam(input_tensor=inputs, targets=targets)
    return grayscale_cams


def initialize_distractor(classifier):
    # target_layers = get_target_layers(classifier)
    # target_layer_params = list(target_layers[0].parameters())
    # cam_shape = (len(target_layers), *target_layer_params[0].shape)  # assuming all target layers have the same shape
    # if finetune.DATASET == 'StanfordCars':  # todo: allow more datasets
    #     output_shape = (360, 240)
    return Distractor()


def train_distractor(distractor, classifier):
    # print(f'distractor: {distractor}')
    # print(f'classifier: {classifier}')
    classifier.eval()
    data_loaders = finetune.get_dataloaders()
    for phase in ['train', 'val']:
        if distractor:
            if phase == 'train':
                distractor.train()
            else:
                distractor.eval()
        for batch in data_loaders[phase]:
            grad_cams = grad_cam_from_batch(classifier, batch)
            # print(grad_cams.shape)
            inputs, labels = batch
            cam_tensors = [torch.Tensor(cam).unsqueeze(0).unsqueeze(0) for cam in grad_cams]
            # print(cam_tensors)
            distractor_outputs = distractor(torch.cat(cam_tensors))
            # below code is purely for visual verification that this works.
            for i in range(grad_cams.shape[0]):
                img = np.float32(inputs[i].permute(1,2,0))
                # print(img.shape)
                # print(grad_cams[i, :].shape)
                low = min(img.flatten())
                high = max(img.flatten())
                img = (img - low) / (high - low)
                visualization = show_cam_on_image(img, grad_cams[i, :], use_rgb=True)
                # print(visualization.shape)
                Image.fromarray(visualization).show()
                # also show masking
                mask = 1 - np.expand_dims(np.float32(grad_cams[i]), -1)
                # mask = 1 - np.float32(distractor_outputs[i].detach().permute(1,2,0))
                print(mask.shape)
                masked_img = np.uint8(np.multiply(img, mask) * 255)
                print(masked_img.shape)
                Image.fromarray(masked_img).show()
            break
        break
