from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

import argparse

learning_rate = 0.00001
batch_size = 16
shuffle = True
pin_memory = True
num_workers = 1
transform = transforms.Compose([
    transforms.PILToTensor(),
])
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class ImageDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def len(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Retrieve image
        img_name = self.annotations.iloc[index, 0]
        img = Image.open(img_name).convert("RGB")
        # Resize image so the width is 224
#        basewidth = 224
#        wpercent = (basewidth/float(img.size[0]))
#        hsize = int((float(img.size[1])*float(wpercent)))
#        img = img.resize((basewidth,hsize), Image.ANTIALIAS)

        img = img.resize((224,224), Image.ANTIALIAS)
        # Add extra dimension
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
        y_label = torch.unsqueeze(y_label, dim=-1)
        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)


class NeuralNet(nn.Module):
    def __init__(self):

        super().__init__()
        # For COCO 256 x 256 images
#        self.conv1 = nn.Conv2d(3, 32, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(32, 64, 5)
#        self.conv3 = nn.Conv2d(64, 128, 5)
#        self.conv4 = nn.Conv2d(128, 128, 5)
#        self.conv5 = nn.Conv2d(128, 64, 5)
#        self.fc1 = nn.Linear(1024, 128)
#        self.fc2 = nn.Linear(128, 10)
#        self.fc3 = nn.Linear(10, 1)


        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=3)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=3)
        self.conv4 = nn.Conv2d(128, 128, 5, stride=2, padding=3)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
#        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def mkdir(indir):
    try:
        os.mkdir(indir)
    except:
        pass

#def run_inference(image_paths, orientations, weights):

def run_inference(image_paths, orientations, net, device):
    n_feats = 1


##    net = NeuralNet()
#    net = torchvision.models.efficientnet_v2_m(pretrained=True,weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT)
##    net = torchvision.models.mobilenet_v3_large(pretrained=True,weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
#    #for param in net.parameters():
#    #    param.requires_grad = False
#    last_item_index = len(net.classifier)-1
#    old_fc = net.classifier.__getitem__(last_item_index )
#    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= n_feats, bias=True)
#    net.classifier.__setitem__(last_item_index , new_fc)


#    checkpoint = torch.load(weights)
#    net.load_state_dict(checkpoint['model_state_dict'])
    input_tensors = []
    for img_name in image_paths:
        img = Image.open(img_name).convert("RGB")
        # Resize image so the width is 224
#        basewidth = 224
#        wpercent = (basewidth/float(img.size[0]))
#        hsize = int((float(img.size[1])*float(wpercent)))
#        img = img.resize((basewidth,hsize), Image.ANTIALIAS)

        img = img.resize((224,224), Image.ANTIALIAS)
        # Add extra dimension
        img = transform(img)
        input_tensors.append(img)
    input_tensors = torch.stack(input_tensors)    
    outputs = net(input_tensors.float().to(device)).tolist()
    orientation_to_count = {}
    for idx,out in enumerate(outputs):
        orientation_to_count[orientations[idx]] = out[0]
    return orientation_to_count

def run_inference_with_weights(image_paths, orientations, weights):
    n_feats = 1

#    net = torchvision.models.efficientnet_v2_m(pretrained=False, weights=None)
#    last_item_index = len(net.classifier)-1
#    old_fc = net.classifier.__getitem__(last_item_index )
#    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= n_feats, bias=True)
#    net.classifier.__setitem__(last_item_index , new_fc)
    net = NeuralNet().to(device)
    checkpoint = torch.load(weights)
    net.load_state_dict(checkpoint['model_state_dict'])
    input_tensors = []
    for img_name in image_paths:
        img = Image.open(img_name).convert("RGB")
        # Resize image so the width is 224

        img = img.resize((224,224), Image.ANTIALIAS)
        # Add extra dimension
        img = transform(img)
        input_tensors.append(img)
    input_tensors = torch.stack(input_tensors)    
    outputs = net(input_tensors.float().to(device)).tolist()
    orientation_to_count = {}
    for idx,out in enumerate(outputs):
        orientation_to_count[orientations[idx]] = out[0]
    return orientation_to_count

def run_inference_with_per_orientation_weights(orientation_to_image_paths, orientation_to_weights):
    n_feats = 1

    orientation_to_count = {}
    for o in orientation_to_image_paths:
        img_name = orientation_to_image_paths[o]
        net = NeuralNet().to(device)
        checkpoint = torch.load(orientation_to_weights[o])
        net.load_state_dict(checkpoint['model_state_dict'])

        img = Image.open(img_name).convert("RGB")
        # Resize image so the width is 224

        img = img.resize((224,224), Image.ANTIALIAS)
        # Add extra dimension
        img = transform(img)
        img = torch.unsqueeze(img, dim=0)
        outputs = net(img.float().to(device))
        orientation_to_count[o] = outputs.item()

        print('Orientation ', o, ' weights file ' , orientation_to_weights[o], ' -> ', orientation_to_count[o])
#        input_tensors = []
#        for img_name in image_paths:
#            img = Image.open(img_name).convert("RGB")
#            # Resize image so the width is 224
#
#            img = img.resize((256,256), Image.ANTIALIAS)
#            # Add extra dimension
#            img = transform(img)
#            input_tensors.append(img)
#        input_tensors = torch.stack(input_tensors)    
#        outputs = net(input_tensors.float().to(device)).tolist()
#        orientation_to_count[o] = outputs[0][0]
    return orientation_to_count

def images_to_annotations_file(training_images, rewards, outfile):
    assert(len(training_images) == len(rewards))
    with open(outfile, 'w') as f:
        for i in range(0,len(training_images)):
            f.write(f'{training_images[i]},{rewards[i]}\n')

def read_annotations(infile):
    training_images = []
    rewards = []
    with open(infile, 'r') as f:
        for line in f.readlines():
            items = line.split(',')
            rewards.append(int(items[1].strip()))
            training_images.append(items[0].strip())
    return training_images, rewards



def train(training_set, training_rewards, val_set, val_rewards,  outdir, project_name, gpu, weights_id=0, max_epochs=30, orientation=None):

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    val_outfile = f'{outdir}/{project_name}_val_min.pt'
    train_outfile = f'{outdir}/{project_name}_train_min.pt'
    net = NeuralNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#    checkpoint = torch.load(current_weights_file)
#    net.load_state_dict(checkpoint['model_state_dict'])
    min_loss = 10000.0

    losses = []
    best_output_list = []
    best_train_loss = 100000.0
    best_val_loss = 100000.0
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_val_loss = 0.0
        output_list = []
        for i in range(0,len(training_set)): 
            img = Image.open(training_set[i]).convert("RGB")
            # Resize image so the width is 224

            img = img.resize((224,224), Image.ANTIALIAS)
            # Add extra dimension
            y_label = torch.tensor(float(training_rewards[i] ))
            y_label = torch.unsqueeze(y_label, dim=-1)
            y_label = torch.unsqueeze(y_label, dim=-1)
            if transform is not None:
                img = transform(img)
            optimizer.zero_grad()
            img = torch.unsqueeze(img, dim=0)
            outputs = net(img.float().to(device))
            loss = criterion(outputs, y_label.to(device))
#            output_list.append((training_set[i], outputs.item(), y_label.item()))
#            total_loss += loss.item()
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i in range(0,len(val_set)): 
                img = Image.open(val_set[i]).convert("RGB")
                # Resize image so the width is 224
                img = img.resize((224,224), Image.ANTIALIAS)
                # Add extra dimension
                y_label = torch.tensor(float(training_rewards[i] ))
                y_label = torch.unsqueeze(y_label, dim=-1)
                y_label = torch.unsqueeze(y_label, dim=-1)
                if transform is not None:
                    img = transform(img)
                img = torch.unsqueeze(img, dim=0)
                outputs = net(img.float().to(device))
                loss = criterion(outputs, y_label.to(device))
                output_list.append((val_set[i], outputs.item(), y_label.item()))

                total_val_loss += loss.item()

        losses.append(total_val_loss)
        if epoch % 2 == 0:
            print('Epoch ', epoch , ' train loss ' , total_train_loss, ' val loss ', total_val_loss)
        if total_train_loss < best_train_loss:
            best_train_loss= total_train_loss
            best_output_list = output_list
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
        #            'loss': LOSS,
                    }, train_outfile)

        if total_val_loss < best_val_loss:
            best_val_loss= total_val_loss
            best_output_list = output_list
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
        #            'loss': LOSS,
                    }, val_outfile)
        if len(losses) > 5:
            avg_loss = sum(losses[len(losses)-5:]) / len(losses[len(losses)-5:]) 
            if avg_loss <= 0.1:
                break
#    print('Sample results') 
#    print('Outputs ', best_output_list)

    if orientation is not None:
        print('Weights file for orientation ', orientation)

    '''
    annotations_file = os.path.join(tmp_dir, f'dataset{weights_id}.csv')
    images_to_annotations_file(training_images, training_rewards, annotations_file)
    print('Written dataset to ', annotations_file)
    outfile = os.path.join(tmp_dir, f'model_cont{weights_id}.pt')
    if orientation is not None:
        outfile = os.path.join(tmp_dir, f'model_cont{weights_id}-{orientation}.pt')
    dataset = ImageDataset("train", annotations_file, transform=transform)

    train_set_size = dataset.len()
    validation_set_size = dataset.len() - train_set_size

    train_set, validation_set = torch.utils.data.random_split(dataset,[train_set_size, validation_set_size])
    shuffle=False
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)


    net = NeuralNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    checkpoint = torch.load(current_weights_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    min_loss = 10000.0
#    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    losses = []
    def train_model( max_epochs, net, criterion, optimizer, losses ):
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            total_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs.float().to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                # print statistics
                total_loss += loss.item()
                print('Outputs ', outputs)
                print('GT ', labels)
            print(f'Epoch {epoch + 1} loss {total_loss}')
            losses.append(total_loss)
            if len(losses) > 5:
                avg_loss = sum(losses[len(losses)-5:]) / len(losses[len(losses)-5:]) 
                if avg_loss <= 0.1:
                    print('Avg loss is ', avg_loss, ' breaking')
                    break

    train_model(max_epochs, net, criterion, optimizer, losses)
    avg_loss = sum(losses[len(losses)-5:]) / len(losses[len(losses)-5:]) 
    if  avg_loss >= 0.5:
        print('Training more ... ')
        train_model(int(7 * avg_loss), net, criterion, optimizer, losses)
        

    torch.save({
            'epoch': max_epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': LOSS,
            }, outfile)
    '''
    return  losses[-1]




def main():
    
    ap = argparse.ArgumentParser()

    ap.add_argument('annotations', type=str, help='File with training labels')
    ap.add_argument('outfile', type=str, help='Output weights file')
    args = ap.parse_args()
    training_images, rewards = read_annotations(args.annotations)
    train(training_images, training_rewards)

