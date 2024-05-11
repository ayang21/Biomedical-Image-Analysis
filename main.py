import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets,transforms, models

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR

# Function to display images
def imshow(inp, titles):
    # If inp and titles are not lists, convert them to lists
    if not isinstance(inp, list):
        inp = [inp]
    if not isinstance(titles, list):
        titles = [titles]

    fig, axs = plt.subplots(1, len(inp), figsize=(5 * len(inp), 5))

    # If there's only one image, convert it to a list
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, ax in enumerate(axs):
        # Convert PyTorch tensor to numpy array and transpose dimensions
        image = inp[i].cpu().numpy().transpose((1, 2, 0))
        # Normalize the numpy array
        image = std * image + mean
        image = np.clip(image, 0, 1)
        # Display the image and set the title
        ax.imshow(image)
        ax.set_title(titles[i])
        # Remove the axis
        ax.axis('off')

    plt.show()

def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, num_epochs=5):
    history = {'train_loss': [], 'train_acc': [], 'validation_loss': [], 'validation_acc': []}

    for epoch in range(num_epochs):
        print('=' * 10)
        print('epoch=',epoch + 1)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = validation_loader

            cum_loss = 0.0
            running_corrects = 0

            for images, labels  in (dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                cum_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)
            
            # # Optional: choose one of these schedulers to use in training 
            # scheduler.step()  # step-based learning scheduler; comment out if needed
            # clr_scheduler.step() # cyclic learning scheduler; comment out if needed
            # cosine_scheduler.step() # cosine annealing scheduler; comment out if needed
            # warmup_scheduler.step() # learning rate warm up scheduler; comment out if needed

            temp_loss = cum_loss / dataset_sizes[phase]
            temp_acc = running_corrects / dataset_sizes[phase]

            print('%s Epoch - %d, loss - %0.5f Acc - %0.5f'\
                %(phase, epoch, temp_loss, temp_acc))

            history[phase+'_loss'].append(temp_loss)
            history[phase+'_acc'].append(temp_acc)

        # # Optionally unfreeze layers after 5 epochs; comment out if needed
        # if epoch >= 2:
        #     for name, child in model.named_children():
        #         if name in ['layer4', 'fc']:
        #             for param in child.parameters():
        #                 param.requires_grad = True
        #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    return model, history

def cross_validate_model(model, criterion, optimizer, scheduler, dataset, num_epochs=5, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)

    # Get the targets of the dataset
    targets = [target for _, target in dataset]

    for fold, (train_index, valid_index) in enumerate(skf.split(np.zeros(len(dataset)), targets)):
        print(f"\nFold {fold+1}")

        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        model, history = train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, num_epochs)
    
    return model, history

file_ext = "png"

base_path = 'tbx11k-DatasetNinja/'
train_path = 'train-data/'
test_path = 'test-data/'
validation_path = 'validation-data/'
normal_destination_path = 'data-normal/'
tb_destination_path = 'data-tb/'
sick_destination_path = 'data-sick/'

test_ann_path = base_path + 'test/ann/'
test_img_path = base_path + 'test/img/'

train_ann_path = base_path + 'train/ann/'
train_img_path = base_path + 'train/img/'

val_ann_path = base_path + 'val/ann/'
val_img_path = base_path + 'val/img/'

# Organize train data into a class strucutre with normal, tb and sick folders
for file in os.listdir(train_ann_path):
    # Remove .json extension
    imgname = file[:-5]

    # Checks if healthy_result (annotation of normal) is in report, if so, copy to normal_destination_path else copy to tb_destination_path
    if imgname[0] == 'h':
        shutil.copy(train_img_path + imgname, train_path + normal_destination_path + imgname)
    elif imgname[0] == 's':
        shutil.copy(train_img_path + imgname, train_path + sick_destination_path + imgname)
    else:
        shutil.copy(train_img_path + imgname, train_path + tb_destination_path + imgname)

# Organize validation data into a class strucutre with normal, tb and sick folders
for file in os.listdir(val_ann_path):
    # Remove .json extension
    imgname = file[:-5]

    # Checks if healthy_result (annotation of normal) is in report, if so, copy to normal_destination_path else copy to tb_destination_path
    if imgname[0] == 'h':
        shutil.copy(val_img_path + imgname, validation_path + normal_destination_path + imgname)
    elif imgname[0] == 's':
        shutil.copy(val_img_path + imgname, validation_path + sick_destination_path + imgname)
    else:
        shutil.copy(val_img_path + imgname, validation_path + tb_destination_path + imgname)

# Organize test data into a class strucutre with unknown
for file in os.listdir(test_ann_path):
    # Remove .json extension
    imgname = file[:-5]

    shutil.copy(test_img_path + imgname, test_path + 'unknown/' + imgname)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load and preprocess data
#Splitting the dataset into train and test using the SubsetSampler class
#To apply two different sets of transforms to train and test, I'm loading the dataset twice.
#I've applied resize so as to make the images smaller and thus model training faster. Models trained on smaller images tend to not overfit and generalize well for larger images.
#I've also added a random crop and random flip to augment the dataset further.

#The mean and standard deviation of the ImageNet data Alexnet was trained on.
mean = [0.485, 0.456, 0.406] # WILL NEED TO CHANGE THIS
std  = [0.229, 0.224, 0.225] # WILL NEED TO CHANGE THIS


test_size = 0.30
random_seed = 24
num_workers = 0
batch_size = 12

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10), #added extra transformation to training; comment out if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)
])
test_transform = transforms.Compose([
    transforms.Resize(256),    
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)
])

# Load the train dataset with transform
train_dataset = datasets.ImageFolder(root=train_path,
                                  transform=train_transform) #replaced dataset_dir with base_path

# Load the test dataset with transform
test_dataset = datasets.ImageFolder(root=test_path,
                                  transform=test_transform) #replaced dataset_dir with base_path

# Load the validation dataset with transform
validation_dataset = datasets.ImageFolder(root=validation_path,
                                         transform=test_transform)


# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)

# Create data loader for validation data
validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                batch_size=batch_size, 
                                                num_workers=num_workers, shuffle=True)

dataloaders = {
    'train': train_loader,
    'test': test_loader,
    'validation': validation_loader
}
dataset_sizes = { 'train': len(train_dataset), 'validation': len(validation_dataset), 'test': len(test_dataset) }
# Explore data set
class_names = train_dataset.classes

# Fetches a batch of inputs and their corresponding classes from the 'train' data loader.
inputs, classes = next(iter(dataloaders['train']))

# Utilizes torchvision's utility to make a grid of images from the batch, which helps in visualizing multiple images simultaneously.
inputs_list = [inputs[i] for i in range(inputs.size(0))]
titles_list = [class_names[classes[i]][5:] for i in range(classes.size(0))]
# Calls the imshow function defined above to display the grid of images with class names as titles. 
imshow(inputs_list, titles_list)

# Finetuning the pretrained model

model = models.alexnet(pretrained=True)
num_ftrs = model.classifier[6].in_features

#Redefining the last layer to classify inputs into the two classes we need as opposed to the original 1000 it was trained for.
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))
criterion   = nn.CrossEntropyLoss()
optimizer   = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)

# # You can use ResNet50 with pretrained weights; comment out if needed
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # Adjust to your number of classes

# # Alternatively, you can use Densenet as the model; comment out if needed
# Initialize DenseNet model
# model = models.densenet121(pretrained=True)
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))

# # Optional 
# for param in model.parameters():
#     param.requires_grad = False  # Initially freeze all parameters

# # Optional: use class weights
# class_weights = torch.tensor([1.0, 1.0, 1.0])  # Initialize with equal weights
# # three classes: normal, sick, and tuberculosis
# # Calculate the inverse of class frequencies as weights
# total_samples = len(train_dataset)
# class_counts = [len([label for _, label in train_dataset if label == i]) for i in range(len(class_weights))]
# class_weights = total_samples / (torch.tensor(class_counts, dtype=torch.float32) + 1e-6)

# # Loss function
# criterion = nn.CrossEntropyLoss() #Optional: can use weight=class_weights.to(device) as parameter if using class weights

# # Can alternatively use Adam optimizer for the classifier
# optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)

# # Optional: Initialize either on of these four schedulers
# # Step-based:
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # Adjust every 3 epochs, reduce lr by a factor of 0.1; comment out if needed
# # Cyclic learning 
# clr_scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000)
# # Cosine annealing
# cosine_scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)
# # Learning rate warm up
# warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1)

#Cross validate the model during training phase to see model's generalization of data
model, history = cross_validate_model(model, criterion, optimizer, scheduler, train_dataset, num_epochs=5) # Change the num_epochs to however many per folds you wish to perform

# model, history = train_model(model, criterion, optimizer, num_epochs=10)

# Plot training and validation loss
plt.figure()
plt.plot(history['train_loss'], label='train')
plt.plot(history['validation_loss'], label='validation')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure()
plt.plot(history['train_acc'], label='train')
plt.plot(history['validation_acc'], label='validation')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Model Evaluation
model.eval() #batchnorm or dropout layers will now work in eval mode instead of training mode.
torch.no_grad() #sets all the requires_grad flag to false and stops all gradient calculation.
accuracies = []
total = 0

for images, labels in dataloaders['test']:

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracies.append(correct / labels.size(0))

overall_accuracy = sum(accuracies) / len(accuracies)
print(f'Overall model accuracy from the test run: {overall_accuracy * 100:.2f}%')

# Plot testing accuracy
plt.figure()
plt.plot(accuracies)
plt.title('Accuracy over test data')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.show()

inputs, labels = next(iter(dataloaders['test']))

inputs = inputs.to(device)
labels = labels.to(device)

inp = torchvision.utils.make_grid(inputs)

outputs = model(inputs)
_, preds = torch.max(outputs, 1)

for j in range(len(inputs)):
    inp = inputs.data[j]
    imshow(inp, 'predicted:' + class_names[preds[j]])

