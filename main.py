import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import datasets,transforms, models


file_ext = "png"
clf_result = "normal"

base_path = 'tbx11k-DatasetNinja/'
normal_destination_path = 'data-normal/'
tb_destination_path = 'data-tb/'

test_ann_path = base_path + 'test/ann/'
test_img_path = base_path + 'test/img/'

train_ann_path = base_path + 'train/ann/'
train_img_path = base_path + 'train/img/'

val_ann_path = base_path + 'val/ann/'
val_img_path = base_path + 'val/img/'

ann_directory = [test_ann_path, train_ann_path, val_ann_path]
img_directory = [test_img_path, train_img_path, val_img_path]

# Collect data

# Skip the validation set for now
for i in range(len(ann_directory) - 1):
    
    # Loop through each file in the directory, going from test to train data
    for file in os.listdir(ann_directory[i]):
         imgname = file[:-len(file_ext)-1]
         with open(os.path.join(ann_directory[i],file)) as report:
             for line in report:
                 if clf_result in line:
                     shutil.copy(img_directory[i] + imgname, normal_destination_path + imgname)
                     break
             else:
                 shutil.copy(img_directory[i] + imgname, tb_destination_path + imgname)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #Load and preprocess data
# #Splitting the dataset into train and test using the SubsetSampler class
# #To apply two different sets of transforms to train and test, I'm loading the dataset twice.
# #I've applied resize so as to make the images smaller and thus model training faster. Models trained on smaller images tend to not overfit and generalize well for larger images.
# #I've also added a random crop and random flip to augment the dataset further.

# #The mean and standard deviation of the ImageNet data Alexnet was trained on.
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225] 
test_size = 0.30
random_seed = 24
num_workers = 0
batch_size = 8
train_transform = transforms.Compose([
     transforms.Resize(256),
     transforms.RandomCrop(224),
     transforms.RandomHorizontalFlip(),
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
train_dataset = datasets.ImageFolder(root=base_path,
                                   transform=train_transform) #replaced dataset_dir with base_path

test_dataset = datasets.ImageFolder(root=base_path,
                                   transform=test_transform) #replaced dataset_dir with base_path
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_size * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            sampler=train_sampler,
                                           num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size, 
                                          sampler=test_sampler,
                                          num_workers=num_workers)
dataloaders = {
    'train': train_loader,
    'test': test_loader
}
# Explore data set
class_names = train_dataset.classes

print(class_names) #output=['test', 'train', 'val']
def imshow(inp, title):
    # Converts a PyTorch tensor 'inp' to a numpy array and transposes the dimensions
    # from (channel, height, width) to (height, width, channel) for display purposes.
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    # Normalizes the numpy array 'inp' using the standard deviation and mean used in
    # the normalization step of the transforms, then clips values to be between 0 and 1.
    # This is necessary to display the image correctly after normalization, which shifts pixel values.
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # Creates a new figure with a specified size (12x6 inches) in which the image will be displayed.
    plt.figure (figsize = (12, 6))
    # Displays the image contained in 'inp'. The title of the image is set to the provided 'title'.
    plt.imshow(inp)
    plt.title(title) 
    # Pauses the display for 5 seconds, allowing the image to be visible for this duration.
    # This is useful in interactive environments where the output might otherwise be too fleeting.
    plt.pause(5)  
# Fetches a batch of inputs and their corresponding classes from the 'train' data loader.
inputs, classes = next(iter(dataloaders['train']))
# Utilizes torchvision's utility to make a grid of images from the batch, which helps in visualizing multiple images simultaneously.
out = torchvision.utils.make_grid(inputs)
# Calls the imshow function defined above to display the grid of images with class names as titles.
imshow(out, title=[class_names[x] for x in classes])
print(classes) #output=tensor([0, 2, 1, 0, 0, 1, 1, 1])
print([class_names[x] for x in classes]) #output=['test', 'val', 'test', 'val', 'val', 'train', 'train', 'test']

# Finetuning the pretrained model

model = models.alexnet(pretrained=True)
model
num_ftrs = model.classifier[6].in_features
num_ftrs

#Redefining the last layer to classify inputs into the two classes we need as opposed to the original 1000 it was trained for.
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))
criterion   = nn.CrossEntropyLoss()

optimizer   = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
def train_model(model, criterion, optimizer, num_epochs=25):

    model = model.to(device)
    total_step = len(dataloaders['train'])


    for epoch in range(num_epochs):
        print('epoch=',epoch)        

        for images, labels  in (dataloaders['train']):

                images = images.to(device)
                labels = labels.to(device)
    
                outputs = model(images)
                outputs = outputs.to(device)
                loss = criterion(outputs,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Epoch - %d, loss - %0.5f '\
            %(epoch, loss.item()))

    return model
model = train_model(model, criterion, optimizer, num_epochs=10)
# Model Evaluation
model.eval() #batchnorm or dropout layers will now work in eval mode instead of training mode.
torch.no_grad() #sets all the requires_grad flag to false and stops all gradient calculation.
correct = 0
total = 0

for images, labels in dataloaders['test']:

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the model on the test images: {}%'\
      .format(100 * correct / total))
inputs, labels = iter(dataloaders['test']).next()

inputs = inputs.to(device)
inp = torchvision.utils.make_grid(inputs)

outputs = model(inputs)
_, preds = torch.max(outputs, 1)

for j in range(len(inputs)):
    print ("Acutal label", np.array(labels)[j])

    inp = inputs.data[j]
    imshow(inp, 'predicted:' + class_names[preds[j]])

