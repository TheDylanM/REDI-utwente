import functools
from typing import List

import torchvision
import torchvision.transforms.functional
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models
from PIL import Image
from torch import optim
import tqdm

import finetune


def scale_cam_image(cam: torch.Tensor, target_size=None) -> torch.Tensor:
    cam = cam - torch.min(cam, 0).values
    cam = cam / (1e-7 + torch.max(cam, 0).values)
    if target_size is not None:
        cam = torchvision.transforms.functional.resize(cam, target_size)
    return cam

def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[torch.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).T
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(0)
        U, S, VT = torch.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection.unsqueeze(0))
    return torch.cat(projections)

class TensorGradCAM(GradCAM):
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return torch.mean(grads, (2, 3))  # use torch.mean instead of np.mean

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> torch.Tensor:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> torch.Tensor:

        activations_list = [a.cpu()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = torch.maximum(cam, torch.tensor(0))
            # todo: reimplement scale_cam_image
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer


class Distractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # todo: consider using a conv layer with 4 -> 3 channels, outputting rgb instead of mask
        self.cnn = torch.nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1)  # preserve dimensions

    def forward(self, x):
        x = self.cnn(x)
        x = functional.relu(x)
        x = functional.softsign(x)
        return x


def get_target_layers(classifier):
    target_layers = []
    if type(classifier) == models.ResNet: # todo: more models
        target_layers = [classifier.layer4[-1]]
    return target_layers


def grad_cam_from_batch(classifier, batch):
    # todo: understand how target_layers argument to gradcam works
    target_layers = get_target_layers(classifier)
    inputs, labels = batch
    inputs = inputs.to(finetune.device)
    with TensorGradCAM(model=classifier, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        targets = [ClassifierOutputTarget(l) for l in labels.tolist()]
        grayscale_cams = cam(input_tensor=inputs, targets=targets)
    cam_tensors = [torch.Tensor(cam).unsqueeze(0).unsqueeze(0).to(finetune.device) for cam in grayscale_cams]
    return torch.cat(cam_tensors)


def initialize_distractor(classifier):
    distractor = Distractor()
    distractor.to(finetune.device)
    return distractor


def cam_avg_std(cam_batch):
    # calculates the average standard deviation of a batch of class activation mappings
    stds = torch.std(cam_batch, tuple(range(1, len(cam_batch.shape))))
    return torch.sum(stds)/stds.nelement()

def train_distractor(distractor, classifier):
    # print(f'distractor: {distractor}')
    # print(f'classifier: {classifier}')
    assert distractor is not None
    assert classifier is not None
    classifier.eval()
    data_loaders = finetune.get_dataloaders()

    params_to_update = distractor.parameters()
    DISTRACTOR_LR = 0.001
    DISTRACTOR_MOMENTUM = finetune.MOMENTUM
    DISTRACTOR_NUM_EPOCHS = 1
    optimizer = optim.SGD(params_to_update, lr=DISTRACTOR_LR, momentum=DISTRACTOR_MOMENTUM)
    criterion = cam_avg_std

    baseline_loss_history = {'train': [], 'val': []}
    distractor_loss_history = {'train': [], 'val': []}
    for epoch in range(DISTRACTOR_NUM_EPOCHS):
        for phase in ['train', 'val']:
            training = phase == 'train'
            if training:
                distractor.train()
            else:
                distractor.eval()
            for batch in tqdm.tqdm(data_loaders[phase]):
                # print(grad_cams.shape)
                optimizer.zero_grad()
                inputs, labels = batch
                inputs = inputs.to(finetune.device)
                cams = grad_cam_from_batch(classifier, (inputs, labels))
                predictions = classifier(inputs)
                distractor_outputs = distractor(cams)
                distracting_inputs = torch.multiply(inputs, 1 - distractor_outputs)  # todo: make sure dimensions work
                distracted_predictions = classifier(distracting_inputs)
                distracted_cams = grad_cam_from_batch(classifier, (distracting_inputs, labels))

                baseline_loss = criterion(cams)
                distractor_loss = criterion(distracted_cams)

                baseline_loss_history[phase].append(baseline_loss)
                distractor_loss_history[phase].append(distractor_loss)
                if training:
                    distractor_loss.requires_grad = True
                    distractor_loss.backward()
                    optimizer.step()
    return baseline_loss_history, distractor_loss_history
