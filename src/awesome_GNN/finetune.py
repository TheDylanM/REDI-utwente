import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.nn import functional
from torchvision import datasets, models, transforms
import time
import os
import copy
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSIFIER_OPTIONS = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
DATASET_OPTIONS = ['StanfordCars', 'FGVC-Aircraft']
OCCLUSION_OPTIONS = [None, '0', '1', 'GAUSSIAN', 'SOFTMAX']

DATA_PATH = '../../data'  # write to this variable when importing this module from different directory context than assumed here
FINETUNED_MODELS_PATH = os.path.join(DATA_PATH,
                                     'finetuned_models')  # Path to save the finedtuned models, relative to the global path
DATASET = 'FGVC-Aircraft'  # write to this variable if you wish to use another dataset

# hyperparams/parameters that need defining or tuning
CLASSIFIER_NAME = 'resnet'
CLASSIFIER_INPUT_SIZE = None
BATCH_SIZE = 8
NUM_EPOCHS = 15
# NUM_CLASSES = 0
FEATURE_EXTRACT = False
OPTIMIZER_NAME = 'adam'
LR = 0.001
MOMENTUM = 0.9
ADAM_LR = 0.0001
ADAM_BETA_ONE = 0.9
ADAM_BETA_TWO = 0.999
ADAM_EPSILON = 0.0000007
OCCLUSION_PROBABILITY = 0.25
OCCLUSION_THRESHOLD = 0.85
OCCLUSION_NAME = None  # name for the type of occlusion used. options are listed above

BASICALLY_INFINITY = 10000


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
        'FGVC-Aircraft': 100,
    }[DATASET]


def get_classifier_input_size():
    return {
        'resnet': 224,
        'alexnet': 224,
        'vgg': 224,
        'squeezenet': 224,
        'densenet': 224,
        'inception': 299,
    }[CLASSIFIER_NAME]


def get_data_transforms():
    channel_distributions = {
        'StanfordCars': {
            'means': [0.4593, 0.4466, 0.4453],
            'stddevs': [0.2919, 0.2910, 0.2988]
        },
        'FGVC-Aircraft': {
            'means': [0.4842, 0.5142, 0.5346],
            'stddevs': [0.2251, 0.2185, 0.2495]
        }
    } # distributions are based on train sets, see rgb_channel_normalization.ipynb
    channel_means = channel_distributions[DATASET]['means']
    channel_stddevs = channel_distributions[DATASET]['stddevs']
    input_size = get_classifier_input_size()
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_stddevs)
    ]
    default_transforms = [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        *final_transforms
    ]
    training_transforms = [
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        *final_transforms

    ]
    default_transforms = transforms.Compose(default_transforms)
    training_transforms = transforms.Compose(training_transforms)
    return {
        'train': training_transforms,
        'val': default_transforms,
        'test': default_transforms,
    }


def get_dataset(subsets):
    # assumes a path pointing to a set of folders representing classes, and the samples within those
    # classes to be in their respective folders
    tforms = get_data_transforms()
    dataset = {}
    for x in subsets:
        dataset[x] = datasets.ImageFolder(os.path.join(dataset_path(), x), tforms[x])
    return dataset


def get_dataloaders(subsets=['train', 'val']):
    ds = get_dataset(subsets)
    dataloaders = {}
    for x in subsets:
        dataloaders[x] = torch.utils.data.DataLoader(ds[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloaders


###### functions involved with finetuning
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name=None, use_pretrained=True, _verbose=True):
    if model_name is None:
        model_name = CLASSIFIER_NAME
    num_classes = get_num_classes()
    feature_extract = FEATURE_EXTRACT
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if _verbose:
        print(f'initializing model: {model_name}')

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


def get_target_layers(classifier):
    target_layers = []
    if type(classifier) == models.ResNet:  # todo: more models
        target_layers = [classifier.layer4[-1]]
    elif type(classifier) == models.DenseNet:
        target_layers = [classifier.features.denseblock4.denselayer16]
    elif type(classifier) == models.VGG:
        # print(classifier)
        target_layers = [classifier.features[-1]]
    return target_layers


def threshold_mask(cams):
    mask = cams < OCCLUSION_THRESHOLD
    return 1 * mask[:, None, :, :]


def softmax_mask(cams):
    # calculates softmax in 2D
    softmask = functional.softmax(cams, dim=-1) * functional.softmax(cams, dim=-2)
    # invert the softmax, such that it can be used as a multiplicative mask
    return 1 * (1 - softmask)[:, None, :, :]


def gaussian_smoothing(inputs):
    # simple symmetric gaussian window, performed as a torchvision transform
    SCALE = 15
    KERNEL_SIZE = 2 * SCALE + 1
    gaussian_filter = torchvision.transforms.GaussianBlur(KERNEL_SIZE, SCALE)
    return gaussian_filter(inputs)


def apply_occlusion(inputs, model):
    target_layers = get_target_layers(model)

    # by using 'with', the cam object is properly opened and closed
    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cams = cam(input_tensor=inputs)
        grayscale_cams = torch.tensor(grayscale_cams).to(device)

    if OCCLUSION_NAME == '0' or OCCLUSION_NAME == '1':
        mask = threshold_mask(grayscale_cams)
        inputs = inputs * mask
        if OCCLUSION_NAME == '1':
            inputs += 1 - mask
    if OCCLUSION_NAME == 'GAUSSIAN':
        mask = threshold_mask(grayscale_cams)
        smoothed_inputs = gaussian_smoothing(inputs)
        inputs = inputs * mask + smoothed_inputs * (1 - mask)
    if OCCLUSION_NAME == 'SOFTMAX':
        mask = softmax_mask(grayscale_cams)
        inputs = inputs * mask
    return inputs


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                checkpoint_save=0,
                num_epochs=NUM_EPOCHS,
                is_inception=False,
                is_retrain=None):
    # use is_retrain when loading from checkpoint, must be int of the last epoch
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1
    if num_epochs is None:
        num_epochs = BASICALLY_INFINITY
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_sample_count = 0

            # Iterate over data.
            tqdm_obj = tqdm.tqdm(dataloaders[phase])
            tqdm_obj.set_description(desc=f'Epoch {epoch}/{num_epochs - 1} {phase}')
            for i, (inputs, labels) in enumerate(tqdm_obj):
                running_sample_count += len(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train' and OCCLUSION_NAME is not None:
                    # Apply occlusion to only half of the batches
                    if np.random.rand(1)[0] <= OCCLUSION_PROBABILITY:
                        inputs = apply_occlusion(inputs, model)

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
                tqdm_obj.set_postfix({
                    # f'phase': phase,
                    'loss': f'{running_loss / running_sample_count:.5g}',
                    'accuracy': f'{running_corrects.double() / running_sample_count:.5g}',
                })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

                # Create checkpoint
                # check for modulo of epoch + 1, because epochs start at 0.
                # saving every 10th epoch means saving at epoch 9, not epoch 10.
                if (epoch + 1) % checkpoint_save == 0 and checkpoint_save != 0:
                    e = epoch + 1
                    if is_retrain:
                        e = epoch + 1 + is_retrain

                    state = {
                        'name': CLASSIFIER_NAME,
                        'epochs': e,
                        'model_state_dict': best_model_wts,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'occlusion': OCCLUSION_NAME,
                        'val_acc_history': val_acc_history,
                        'train_acc_history': train_acc_history,
                        'train_loss_history': train_loss_history
                    }

                    # save model
                    safe_mkdir(os.path.join(FINETUNED_MODELS_PATH, DATASET, CLASSIFIER_NAME))

                    save_model(state)
        # early stopping
        if num_epochs >= BASICALLY_INFINITY and epoch - best_epoch >= 30:
            break

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
        'occlusion': OCCLUSION_NAME,
        'val_acc_history': val_acc_history,
        'train_acc_history': train_acc_history,
        'train_loss_history': train_loss_history
    }
    return model, val_acc_history, state


def finetune_model(model,
                   checkpoint_save: int = 10,
                   optimizer_state_dict=None,
                   _verbose=False,
                   is_retrain=None):
    # Optimizer can be: [SGD, Adam]
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

    if OPTIMIZER_NAME == 'adam':
        optimizer = optim.Adam(params_to_update, lr=ADAM_LR, betas=(ADAM_BETA_ONE, ADAM_BETA_TWO), eps=ADAM_EPSILON)
    elif OPTIMIZER_NAME == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=LR, momentum=MOMENTUM)
    else:
        raise f"Unrecognized optimizer name '{OPTIMIZER_NAME}'"

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    criterion = nn.CrossEntropyLoss()
    model, hist, state = train_model(model,
                                     get_dataloaders(),
                                     criterion,
                                     optimizer,
                                     checkpoint_save=checkpoint_save,
                                     num_epochs=NUM_EPOCHS,
                                     is_inception=is_inception(),
                                     is_retrain=is_retrain)
    return model, hist, state


def test_model(path, _verbose=False):
    state = load_checkpoint(path)

    # Read properties from the loaded checkpoint state
    model_name = state['name']
    epochs = state['epochs']
    occlusion = state['occlusion']

    if _verbose:
        print(f'name:      {model_name} \n'
              f'epochs:    {epochs} \n'
              f'occlusion: {occlusion}\n'
              f'--------------------------')

    # Load model
    model = get_model_architecture(model_name)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    test_dataloader = get_dataloaders(['test'])['test']

    total_predictions = []
    total_labels = []
    start_time = time.time()

    # Iterate over data.
    # tqdm_obj = tqdm.tqdm(test_dataloader)
    # tqdm_obj.set_description(desc=f'Epoch {epoch}/{num_epochs - 1} {phase}')

    # Testing loop
    for (inputs, labels) in tqdm.tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        total_predictions = np.append(total_predictions, preds.cpu().numpy())
        total_labels = np.append(total_labels, labels.cpu().numpy())

    # Save these metrics, including the predicted and labels with it
    metrics = calculate_metrics(total_predictions, total_labels)
    metrics['total_predicted'] = total_predictions
    metrics['total_labels'] = total_labels

    time_elapsed = time.time() - start_time

    if _verbose:
        print('Elapsed time:', time_elapsed)

    return metrics


def save_model(state):
    # this does assume that the global variables have the appropriate values. So save your model before
    # setting up global variables for a different run of finetuning!
    file_path = format_model_path(CLASSIFIER_NAME, DATASET, state['epochs'])
    print('[CHECKPOINT]', file_path)
    torch.save(state, file_path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu')


def get_information_from_checkpoint(checkpoint, plot=False, figsize=(14, 6)):
    # Checkpoint is the saved state of a model
    train_acc_history = [t.item() for t in checkpoint['train_acc_history']]
    train_loss_history = checkpoint['train_loss_history']  # The loss apparently is not a tensor
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


def format_model_path(name, dataset, epoch, occlusion_name=None):
    if occlusion_name is None:
        occlusion_name = OCCLUSION_NAME
    path = os.path.join(FINETUNED_MODELS_PATH, dataset, name, '')
    if occlusion_name is None:
        return path + str('{}_{}_E{}.pth'.format(name, dataset, epoch))
    else:
        return path + str('{}_{}_E{}_occ{}.pth'.format(name, dataset, epoch, occlusion_name))


def get_model_architecture(name, _verbose=False):
    return initialize_model(model_name=name, use_pretrained=True, _verbose=_verbose)


def write_to_file(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()


def read_file(path):
    return open(path, 'r').read()


def calculate_metrics(preds, labels):
    return {
        'accuracy': accuracy(preds, labels),
        'precision': precision(preds, labels),
        'recall': recall(preds, labels),
        'f1': f1(preds, labels)
    }


def accuracy(preds, labels):
    return (preds == labels).sum() / len(preds)


def precision(preds, labels):
    return precision_score(labels, preds, average='weighted', zero_division=0)


def recall(preds, labels):
    return recall_score(labels, preds, average='weighted', zero_division=0)


def f1(preds, labels):
    return f1_score(labels, preds, average='weighted', zero_division=0)
