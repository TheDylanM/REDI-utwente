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

# hyperparameters
DISTRACTOR_LR = finetune.LR
DISTRACTOR_MOMENTUM = finetune.MOMENTUM
DISTRACTOR_NUM_EPOCHS = 5


# grad cam reimplemented with torch tensors, instead of numpy ndarrays, to allow for backpropagation
def scale_cam_image(cam: torch.Tensor, target_size=None) -> torch.Tensor:
    # this scaling is not really necessary for our purposes, since we don't need the CAM values to be between 0 and 1.
    # cam = cam - torch.min(cam, 0).values  # todo: this may not be exactly optimal for backprop
    # cam = cam / (1e-7 + torch.max(cam, 0).values)  # todo: this may not be exactly optimal for backprop
    if target_size is not None:
        cam = torchvision.transforms.functional.resize(cam, target_size)
    return cam


def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[torch.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = activations.reshape(
            activations.shape[0], -1).T
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(0)
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
        return torch.mean(grads, (2, 3))

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
            print(eigen_smooth)
        else:
            cam = torch.sum(weighted_activations, 1)
        return cam

    def aggregate_multi_layers(self, cam_per_target_layer: List[torch.Tensor]) -> torch.Tensor:
        cam_per_target_layer = torch.concat(cam_per_target_layer, 1)
        cam_per_target_layer = functional.relu(cam_per_target_layer)  # maximum between tensor and 0
        result = torch.mean(cam_per_target_layer, 1)
        return scale_cam_image(result)

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> List[torch.Tensor]:

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
            cam = functional.relu(cam)  # maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> torch.Tensor:
        input_tensor = input_tensor.to(finetune.device)
        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
        outputs = self.activations_and_grads(input_tensor)
        # deviation from library implementation: assume targets are not None
        assert targets is not None, 'targets must not be None'
        assert self.uses_gradients, 'TensorGradCAM is not implemented for uses_gradients=False'
        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


class Distractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # todo: consider using a conv layer with 4 -> 3 channels, outputting rgb instead of mask
        self.cnn = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  # preserve dimensions

    def forward(self, x):
        x = self.cnn(x)
        x = functional.relu(x)
        x = functional.softsign(x)
        return x


class GumbelDistractor(torch.nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1) for _ in range(n_layers)])

    def forward(self, x):
        for cnn in self.cnns:
            x = cnn(x)
        x_soft = x
        x_hard = ((x > 0) * 1 - (x <= 0) * 1 + 1) / 2
        x_gumbel = x_hard - x_soft.detach() + x_soft
        return x_gumbel


def get_target_layers(classifier):
    target_layers = []
    if type(classifier) == models.ResNet:  # todo: more models
        target_layers = [classifier.layer4[-1]]
    return target_layers


def grad_cam_from_batch(classifier, batch):
    target_layers = get_target_layers(classifier)
    inputs, labels = batch
    inputs = inputs.to(finetune.device)
    with TensorGradCAM(model=classifier, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        targets = [ClassifierOutputTarget(l) for l in labels.tolist()]
        grayscale_cams = cam(input_tensor=inputs, targets=targets)
    cam_tensors = [torch.Tensor(cam).unsqueeze(0).unsqueeze(0).to(finetune.device) for cam in grayscale_cams]
    cam_tensors = torch.concat(cam_tensors)
    return cam_tensors


def initialize_distractor(classifier):
    distractor = GumbelDistractor(n_layers=1)
    distractor.to(finetune.device)
    return distractor


def avg(arr):
    return sum(arr) / len(arr)


def cam_avg_std(cam_batch):
    # calculates the average standard deviation of a batch of class activation mappings
    stds = torch.std(cam_batch, tuple(range(1, len(cam_batch.shape))))
    return torch.mean(stds)


def train_distractor(distractor, classifier):
    # print(f'distractor: {distractor}')
    # print(f'classifier: {classifier}')
    assert distractor is not None
    assert classifier is not None
    classifier.eval()
    data_loaders = finetune.get_dataloaders()

    params_to_update = list(distractor.parameters())

    # optimizer = optim.SGD(params_to_update, lr=DISTRACTOR_LR, momentum=DISTRACTOR_MOMENTUM)
    optimizer = optim.Adam(params_to_update, lr=DISTRACTOR_LR)
    criterion = cam_avg_std
    history = []
    baseline = 'baseline'
    distracted = 'distracted'
    avg_parameter_grad = 'avg_parameter_grad'
    avg_parameter_value = 'avg_parameter_value'
    for epoch in range(DISTRACTOR_NUM_EPOCHS):
        history.append({})
        for phase in ['train', 'val']:
            training = phase == 'train'
            history[epoch][phase] = {}
            history[epoch][phase]['baseline'] = []
            history[epoch][phase]['distracted'] = []
            if training:
                distractor.train()
                history[epoch][phase]['avg_parameter_grad'] = []
            else:
                distractor.eval()
            tqdm_obj = tqdm.tqdm(data_loaders[phase])
            description = f'epoch: {epoch}, {phase}'
            tqdm_obj.set_description(desc=description, refresh=False)
            for i, batch in enumerate(tqdm_obj):
                optimizer.zero_grad()
                classifier.zero_grad()
                inputs, labels = batch
                inputs = inputs.to(finetune.device)
                cams = grad_cam_from_batch(classifier, (inputs, labels))
                # predictions = classifier(inputs)
                distractor_outputs = distractor(cams)
                distracting_inputs = torch.multiply(inputs, distractor_outputs)  # todo: make sure dimensions work
                # distracted_predictions = classifier(distracting_inputs)
                distracted_cams = grad_cam_from_batch(classifier, (distracting_inputs, labels))

                baseline_loss = criterion(cams)
                distractor_loss = criterion(distracted_cams)

                history[epoch][phase]['baseline'].append(baseline_loss.cpu().item())
                history[epoch][phase]['distracted'].append(distractor_loss.cpu().item())
                if training:
                    distractor_loss.requires_grad = True
                    torch.nn.utils.clip_grad_norm_(params_to_update, 1)
                    parameter_avg_grad = torch.mean(
                        torch.concat([param.grad.flatten() for param in params_to_update], 0))
                    parameter_avg_value = torch.mean(torch.concat([param.flatten() for param in params_to_update]))
                    history[epoch][phase]['avg_parameter_grad'].append(parameter_avg_grad.cpu().item())
                    optimizer.step()
                tqdm_obj.set_postfix({
                    'avg_baseline_loss': f'{str(avg(history[epoch][phase][baseline])):.6}',
                    'avg_distractor_loss': f'{str(avg(history[epoch][phase][distracted])):.6}',
                    **({
                           'avg_parameter_grad': f'{f"{avg(history[epoch][phase][avg_parameter_grad]):+}":.7}',
                           'current_parameter_grad': f'{f"{parameter_avg_grad.item():+}":.7}',
                           'avg_parameter_value': f'{f"{parameter_avg_value.item():+}":.7}',
                       }
                       if training
                       else {}),
                })
    return history
