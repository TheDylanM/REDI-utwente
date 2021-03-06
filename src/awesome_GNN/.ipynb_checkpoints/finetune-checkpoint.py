import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import tqdm
import matplotlib.pyplot as plt

CLASSIFIER_OPTIONS = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
DATASET_OPTIONS = ['StanfordCars', 'FGVC-Aircraft']  # todo: add options

DATA_PATH = '../../data'  # write to this variable when importing this module from different directory context than assumed here
FINETUNED_MODELS_PATH = os.path.join(DATA_PATH,
                                     'finetuned_models')  # Path to save the finedtuned models, relative to the global path
DATASET = 'StanfordCars'  # write to this variable if you wish to use another dataset

# hyperparams/parameters that need defining or tuning
CLASSIFIER_NAME = 'resnet'
CLASSIFIER_INPUT_SIZE = None
BATCH_SIZE = 8
NUM_EPOCHS = 15
# NUM_CLASSES = 0
FEATURE_EXTRACT = True
LR = 0.001
MOMENTUM = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_hypers():
    # print hyperparameters
    print(
        f'classifier input size: {CLASSIFIER_INPUT_SIZE}\nbatch size: {BATCH_SIZE}\nnum epochs: {NUM_EPOCHS}\nfeature extract: {FEATURE_EXTRACT}')


###### getter functions

def is_inception():
    return CLASSIFIER_NAME == 'inception'


def dataset_folder():
    return {
        'StanfordCars': os.path.join('StanfordCars', 'pytorch_structured_dataset'),
        'FGVC-Aircraft': os.path.join('FGVC-Aircraft', 'pytorch_structured_dataset'),
        # if desirable, more datasets can be listed here
    }[DATASET]


def data_path():
    return DATA_PATH


def dataset_path():
    if not DATASET in DATASET_OPTIONS:
        raise Exception(f'Invalid dataset name "{DATASET}".\nPlease use one of the following: {DATASET_OPTIONS}')
    return os.path.join(data_path(), dataset_folder())


def get_num_classes():
    return {
        'StanfordCars': 196,
        'FGVC-Aircraft': 30,
    }[DATASET]


def get_data_transforms():
    # todo if relevant: adapt this per dataset?
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(CLASSIFIER_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(CLASSIFIER_INPUT_SIZE),
            transforms.CenterCrop(CLASSIFIER_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


def get_dataset():
    # assumes a path pointing to a set of folders representing classes, and the samples within those
    # classes to be in their respective folders
    tforms = get_data_transforms()
    dataset = {}
    for x in ['train', 'val']:
        dataset[x] = datasets.ImageFolder(os.path.join(dataset_path(), x), tforms[x])
    return dataset


def get_dataloaders():
    ds = get_dataset()
    dataloaders = {}
    for x in ['train', 'val']:
        dataloaders[x] = torch.utils.data.DataLoader(ds[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloaders


###### functions involved with finetuning
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name=CLASSIFIER_NAME, use_pretrained=True, _verbose=True):
    num_classes = get_num_classes()
    feature_extract = FEATURE_EXTRACT
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if not model_name in CLASSIFIER_OPTIONS:
        raise Exception(
            f'Invalid classifier name "{model_name}".\nPlease use one of the following: {CLASSIFIER_OPTIONS}')
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        if _verbose:
            print("Invalid model name, exiting...")
        exit()
    global CLASSIFIER_INPUT_SIZE
    CLASSIFIER_INPUT_SIZE = input_size
    model_ft.to(device)
    return model_ft


def train_model(model, dataloaders, criterion, optimizer, checkpoint_save=0, num_epochs=25, is_inception=False,
                is_retrain=None):
    # use is_retrain when loading from checkpoint, must be int of the last epoch
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            # Create checkpoint
            if epoch % checkpoint_save == 0 and checkpoint_save != 0 and epoch != 0:
                e = epoch
                if is_retrain:
                    e = epoch + is_retrain

                state = {
                    'name': CLASSIFIER_NAME,
                    'epochs': e,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc_history': val_acc_history,
                    'train_acc_history': train_acc_history,
                    'train_loss_history': train_loss_history
                }

                # save model
                model_save_path = format_model_path(CLASSIFIER_NAME,
                                                    DATASET,
                                                    state['epochs'])
                
                safe_mkdir(os.path.join(FINETUNED_MODELS_PATH, DATASET, CLASSIFIER_NAME))
                save_model(state, model_save_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    e = num_epochs
    if is_retrain:
        e = num_epochs + is_retrain

    # This state dict is used for the saving/loading models
    state = {
        'name': CLASSIFIER_NAME,
        'epochs': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc_history': val_acc_history,
        'train_acc_history': train_acc_history,
        'train_loss_history': train_loss_history
    }
    return model, val_acc_history, state


def finetune_model(model, checkpoint_save: int = 10, optimizer_state_dict=None, _verbose=False, is_retrain=None):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    # only parameters with requires_grad are actually changed,
    # so if we are finetuning, we will be updating all parameters. if we are doing feature
    # extraction, we will only be updating certain parameters.
    if _verbose:
        print("Params to learn:")
    if FEATURE_EXTRACT:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if _verbose:
                    print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                if _verbose:
                    print("\t", name)

    # todo: choose a different optimizer?
    optimizer = optim.SGD(params_to_update, lr=LR, momentum=MOMENTUM)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    criterion = nn.CrossEntropyLoss()

    model, hist, state = train_model(model, get_dataloaders(),
                                     criterion,
                                     optimizer,
                                     checkpoint_save=checkpoint_save,
                                     num_epochs=NUM_EPOCHS,
                                     is_inception=is_inception(),
                                     is_retrain=is_retrain)
    return model, hist, state


def save_model(state, file_path):
    print('[CHECKPOINT]', file_path)
    torch.save(state, file_path)


def load_checkpoint(path):
    return torch.load(path)


def get_information_from_checkpoint(checkpoint, plot=False, figsize=(14, 6)):
    # Checkpoint is the saved state of a model
    train_acc_history = [t.item() for t in checkpoint['train_acc_history']]
    train_loss_history = checkpoint['train_loss_history'] # The loss apparently is not a tensor
    val_acc_history = [t.item() for t in checkpoint['val_acc_history']]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].plot(train_loss_history)
        axes[1].plot(train_acc_history)
        axes[2].plot(val_acc_history, color='orange')
    return train_acc_history, train_loss_history, val_acc_history


def structure_checkpoints(_verbose=False):
    # Create directory structure for the finetuned models
    # The sub folders are based on the datasets
    main_folder = FINETUNED_MODELS_PATH
    sub_folders = DATASET_OPTIONS
    sub_sub_folders = CLASSIFIER_OPTIONS

    main_folder_path = os.path.join(main_folder)
    safe_mkdir(main_folder_path, _verbose=_verbose)

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(main_folder_path, sub_folder)
        safe_mkdir(sub_folder_path, _verbose=_verbose)
        for sub_sub_folder in sub_sub_folders:
            sub_sub_folder_path = os.path.join(main_folder_path, sub_folder, sub_sub_folder)
            safe_mkdir(sub_sub_folder_path, _verbose=_verbose)


def safe_mkdir(path, _verbose=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if _verbose:
            print('cannot safely create', path)


def format_model_path(name, dataset, epoch):
    return os.path.join(FINETUNED_MODELS_PATH, dataset, name, str('{}_{}_E{}.pth'.format(name, dataset, epoch)))


def get_model_architecture(name, _verbose=False):
    return initialize_model(name, use_pretrained=False, _verbose=_verbose)
