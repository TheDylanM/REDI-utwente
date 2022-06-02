from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

import finetune

def grad_cam_from_batch(classifier, batch):
    # todo: understand how target_layers argument to gradcam works
    target_layers = []
    if type(classifier) == models.ResNet: # todo: more models
        target_layers = [classifier.layer4[-1]]
    inputs, labels = batch
    with GradCAM(model=classifier, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        targets = [ClassifierOutputTarget(l) for l in labels.tolist()]
        grayscale_cams = cam(input_tensor=inputs, targets=targets)
    return grayscale_cams


def initialize_distractor():
    # todo: properly implement
    pass


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
            print(grad_cams.shape)
            # below code is purely for visual verification that this works.
            inputs, labels = batch
            for i in range(grad_cams.shape[0]):
                img = np.float32(inputs[i].permute(1,2,0))
                print(img.shape)
                print(grad_cams[i, :].shape)
                low = min(img.flatten())
                high = max(img.flatten())
                img = (img - low) / (high - low)
                visualization = show_cam_on_image(img, grad_cams[i, :], use_rgb=True)
                Image.fromarray(visualization).show()
                break
            break

