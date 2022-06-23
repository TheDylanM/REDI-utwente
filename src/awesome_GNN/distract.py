import math
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
import torchvision
import torchvision.models as models
import torchvision.transforms.functional
import tqdm
from pytorch_grad_cam import GradCAM, ActivationsAndGradients
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import optim

import finetune

# hyperparameters
DISTRACTOR_LR = 0.01
DISTRACTOR_MOMENTUM = finetune.MOMENTUM
DISTRACTOR_NUM_EPOCHS = 15
CAM = None


# grad cam reimplemented with torch tensors, instead of numpy ndarrays, to allow for backpropagation
def scale_cam_image(cam: torch.Tensor, target_size=None) -> torch.Tensor:
    # this scaling is not really necessary for our purposes, since we don't need the CAM values to be between 0 and 1.
    cam = cam - torch.min(cam, 0).values  # todo: this may not be exactly optimal for backprop
    cam = cam / (1e-7 + torch.max(cam, 0).values)  # todo: this may not be exactly optimal for backprop
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

class DifferentiableActivationsAndGradients(ActivationsAndGradients):

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad] + self.gradients
        output.register_hook(_store_grad)

class DifferentiableGradCAM(GradCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = DifferentiableActivationsAndGradients(
            self.model, target_layers, reshape_transform)


    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return torch.mean(grads, (-2, -1))

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
        # weighted_activations = activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
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

        activations_list = self.activations_and_grads.activations
        grads_list = self.activations_and_grads.gradients
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
            # todo: decide whether or not to use the relu below
            # cam = functional.relu(cam)  # maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> torch.Tensor:
        # input_tensor = input_tensor.to(finetune.device)
        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
        outputs = self.activations_and_grads(input_tensor)
        # deviation from library implementation: assume targets are not None
        assert targets is not None, 'targets must not be None'
        assert self.uses_gradients, 'DifferentiableGradCAM is not implemented for uses_gradients=False'
        # if self.uses_gradients:
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
        x = torch.multiply(functional.softmax(x, dim=-1), functional.softmax(x, dim=-2))
        print(torch.sum(x))
        return 1 - x


class GumbelDistractor(torch.nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1) for _ in range(n_layers)])

    def forward(self, x):
        for cnn in self.cnns:
            x = cnn(x)
        x_soft = torch.multiply(functional.softmax(x, dim=-1), functional.softmax(x, dim=-2))
        # n = x.numel() / finetune.BATCH_SIZE
        n = x.size(dim=-1) * x.size(dim=-2)
        # n = n*2
        x_hard = 1.0 * (x_soft > 1.5 / n)
        x_gumbel = x_hard - x_soft.detach() + x_soft
        # print(f'sum of x_gumbel: {torch.sum(x_gumbel)}')
        # print(f'numel of x: {n}')
        return 1 - x_gumbel


def get_target_layers(classifier):
    target_layers = []
    if type(classifier) == models.ResNet:  # todo: more models
        target_layers = [classifier.layer4[-1]]
    return target_layers


def grad_cam_from_batch(classifier, batch):
    inputs, labels = batch
    inputs = inputs.to(finetune.device)
    targets = [ClassifierOutputTarget(l) for l in labels.tolist()]
    grayscale_cams = CAM(input_tensor=inputs, targets=targets)
    cam_tensors = [cam.unsqueeze(0).unsqueeze(0) for cam in grayscale_cams]
    cam_tensors = torch.concat(cam_tensors)
    return cam_tensors


def initialize_distractor(classifier):
    distractor = GumbelDistractor(n_layers=1)
    # distractor = Distractor()
    distractor.to(finetune.device)
    return distractor


def avg(arr):
    return sum(arr) / len(arr)


def cam_avg_std(cam_batch):
    # calculates the average standard deviation of a batch of class activation mappings
    stds = torch.std(cam_batch, tuple(range(1, len(cam_batch.shape))))
    return torch.mean(stds)


def plot_loss_history(history):
    for phase in ['train', 'val']:
        avg_loss_baseline = np.array(
            [np.mean(list(map(lambda l: l, epoch[phase]['baseline']))) for epoch in history])
        avg_loss_distracted = np.array(
            [np.mean(list(map(lambda l: l, epoch[phase]['distracted']))) for epoch in history])
        loss_differences = np.divide(avg_loss_distracted - avg_loss_baseline, avg_loss_baseline) * 100
        plt.plot(loss_differences)
    plt.gca().invert_yaxis()
    plt.show()


def train_distractor(distractor, classifier):
    # print(f'distractor: {distractor}')
    # print(f'classifier: {classifier}')
    assert distractor is not None
    assert classifier is not None
    classifier.eval()
    # classifier_copy = copy.deepcopy(classifier)
    data_loaders = finetune.get_dataloaders()

    params_to_update = list(distractor.parameters())

    # optimizer = optim.SGD(params_to_update, lr=DISTRACTOR_LR, momentum=DISTRACTOR_MOMENTUM)
    optimizer = optim.Adam(params_to_update, lr=DISTRACTOR_LR)
    criterion = cam_avg_std
    history = []
    baseline = 'baseline'
    distracted = 'distracted'
    avg_parameter_grad = 'avg_parameter_grad'

    target_layers = get_target_layers(classifier)
    with DifferentiableGradCAM(model=classifier, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        global CAM
        CAM = cam
        print(DISTRACTOR_NUM_EPOCHS)
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
                    classifier.zero_grad()
                    inputs, labels = batch
                    inputs = inputs.to(finetune.device)
                    # inputs.requires_grad = True
                    cams = grad_cam_from_batch(classifier, (inputs, labels))
                    # cams = torch.randn(cams.shape, dtype=torch.float32, device=finetune.device)
                    ## predictions = classifier(inputs)
                    distractor_outputs = distractor(cams)
                    # print(distractor_outputs)
                    distracting_inputs = torch.multiply(inputs, distractor_outputs)
                    # print(torch.sum(distractor_outputs))
                    # print(torch.mean(distracting_inputs - inputs)/torch.mean(inputs))
                    # distracting_inputs = torch.multiply(torch.ones(inputs.shape, dtype=torch.float32, device=finetune.device), distractor_outputs)
                    # distracted_cams = distractor_outputs
                    ## distracted_predictions = classifier(distracting_inputs)
                    distracted_cams = grad_cam_from_batch(classifier, (distracting_inputs, labels))
                    # print(torch.all(cams == distracted_cams))

                    baseline_loss = criterion(cams)
                    distracted_loss = criterion(distracted_cams)

                    history[epoch][phase][baseline].append(baseline_loss.cpu().item())
                    history[epoch][phase][distracted].append(distracted_loss.cpu().item())
                    if training:
                        optimizer.zero_grad()
                        assert distracted_loss.requires_grad, "something's amiss here! the loss does not require_grad"
                        distracted_loss.backward()
                        # torch.nn.utils.clip_grad_norm_(params_to_update, 1)
                        optimizer.step()
                        parameter_avg_grad = torch.mean(
                            torch.concat([param.grad.flatten() for param in params_to_update], 0))
                        parameter_avg_value = torch.mean(torch.concat([param.flatten() for param in params_to_update]))
                        history[epoch][phase]['avg_parameter_grad'].append(parameter_avg_grad.cpu().item())

                    tqdm_obj.set_postfix({
                        'avg_baseline_loss': f'{avg(history[epoch][phase][baseline]):.5g}',
                        'avg_distractor_loss': f'{avg(history[epoch][phase][distracted]):.5g}',
                        **({
                               'avg_parameter_grad': f'{avg(history[epoch][phase][avg_parameter_grad]):+.5g}',
                               'current_parameter_grad': f'{parameter_avg_grad.item():+.5g}',
                               'avg_parameter_value': f'{parameter_avg_value.item():+.5g}',
                           }
                           if training
                           else {}),
                    })
            # plot_loss_history(history)
    return history
