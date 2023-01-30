
# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml

from backbone import EfficientDetBackbone, EfficientNetCounter
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
import torch.nn as nn
import torch.optim as  optim
import random
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--inputs', type=str, help='Input txt file with full names of files to run inference on')
ap.add_argument('-v', '--validation', type=str, help='Input txt file with full names of files to run inference on for validation')
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')

ap.add_argument('-b', '--backbone-weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')

ap.add_argument('-s', '--saved-path', type=str, default=None, help='/path/to/weights')

ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=1)
ap.add_argument('--float16', type=boolean_string, default=False)
args = ap.parse_args()
input_file = args.inputs

compound_coef = 0
saved_path = args.saved_path
val_file = args.validation
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
project_name = args.project
bifpn_weights = None
backbone_weights = 'weights/efficientnet-d0-backbone.pth' if args.backbone_weights is None else args.backbone_weights

bifpn_weights = 'weights/efficientnet-d0-bifpn.pth' if bifpn_weights is None else bifpn_weights


params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

def continual_train(params, input_file, val_file, weights_path,   saved_path,   num_epochs=10):

    img_to_labels = {}

    model.load_state_dict(torch.load(weights_path , map_location={'cuda:0':'cuda:1'} ),strict=False)
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            items = line.strip().split(',')
            frame = int(items[0])
            orientation = items[1]
            image_path = items[2]
            gt_car_count = float(items[3])
            gt_person_count = float(items[4])
            img_to_labels[image_path] = torch.tensor([gt_car_count, gt_person_count])
#            img_to_labels[image_path] = torch.tensor([gt_car_count])
            img_to_labels[image_path] = torch.tensor([gt_person_count])

    img_to_val_labels = {}
    with open(val_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            items = line.strip().split(',')
            frame = int(items[0])
            orientation = items[1]
            image_path = items[2]
            gt_car_count = float(items[3])
            gt_person_count = float(items[4])
            img_to_val_labels[image_path] = torch.tensor([gt_car_count, gt_person_count])
#            img_to_val_labels[image_path] = torch.tensor([gt_car_count])
#            img_to_val_labels[image_path] = torch.tensor([gt_person_count])

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    all_train_losses = []
    all_val_losses = []
    min_loss = 100000.0
    best_weights_path = weights_path
    # In format frame,orientation,file
    for epoch in range(0, num_epochs):
        total_loss = 0.0
        print('EPOCCH ', epoch)
        for image_path in img_to_labels:
            labels = img_to_labels[image_path]
#            print('Processing ', image_path)
            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
            x = torch.from_numpy(framed_imgs[0])
            if use_cuda:
                labels = labels.cuda(gpu)
                x = x.cuda(gpu)
                if use_float16:
                    x = x.half()
                else:
                    x = x.float()
            else:
                x = x.float()

            x = x.unsqueeze(0).permute(0, 3, 1, 2)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss
        # Validation testing
        with torch.no_grad():
            for image_path in img_to_val_labels:
                labels = img_to_val_labels[image_path]
                ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
                x = torch.from_numpy(framed_imgs[0])
                if use_cuda:

                    labels = labels.cuda(gpu)
                    x = x.cuda(gpu)
                    if use_float16:
                        x = x.half()
                    else:
                        x = x.float()
                else:
                    x = x.float()

                x = x.unsqueeze(0).permute(0, 3, 1, 2)
                outputs = model(x)

                loss = criterion(outputs, labels)
                val_loss += loss
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(saved_path, f'efficientnet-d0-counter_min.pth'))
            best_weights_path = os.path.join(saved_path, f'efficientnet-d0-counter_min.pth')

    return best_weights_path

def train(input_file, val_file,  model, saved_path, params ):


    img_to_labels = {}

    with open(input_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            items = line.strip().split(',')
            frame = int(items[0])
            orientation = items[1]
            image_path = items[2]
            gt_car_count = float(items[3])
            gt_person_count = float(items[4])
            img_to_labels[image_path] = torch.tensor([gt_car_count, gt_person_count ])
#            img_to_labels[image_path] = torch.tensor([gt_car_count])
#            img_to_labels[image_path] = torch.tensor([gt_person_count])

    img_to_val_labels = {}
    with open(val_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            items = line.strip().split(',')
            frame = int(items[0])
            orientation = items[1]
            image_path = items[2]
            gt_car_count = float(items[3])
            gt_person_count = float(items[4])
            img_to_val_labels[image_path] = torch.tensor([gt_car_count , gt_person_count ])
#            img_to_val_labels[image_path] = torch.tensor([gt_car_count])
#            img_to_val_labels[image_path] = torch.tensor([gt_person_count])

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    criterion = nn.MSELoss()

#    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    all_train_losses = []
    all_val_losses = []
    min_loss = 100000.0
    max_epochs = 100
    # In format frame,orientation,file
    for epoch in range(0, max_epochs):
        total_loss = 0.0
        num_losses = 0
        print('EPOCCH ', epoch)
        batch_inputs = []
        batch_labels = []
        for image_path in img_to_labels:
            labels = img_to_labels[image_path]
#            print('Processing ', image_path)
            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
            x = torch.from_numpy(framed_imgs[0])
            if use_cuda:
                labels = labels.cuda(gpu)
                x = x.cuda(gpu)
                if use_float16:
                    x = x.half()
                else:
                    x = x.float()
            else:
                x = x.float()

            x = x.unsqueeze(0).permute(0, 3, 1, 2)

            batch_inputs.append(x)
            batch_labels.append(labels)

            if len(batch_inputs) >= 8:
                optimizer.zero_grad()
                x_input = torch.stack(batch_inputs).squeeze(0)
                print('reg input ', x.shape)
                print('modified input ', x_input.shape)
                outputs = model(x_input)

                loss = criterion(outputs, torch.stack(batch_labels).squeeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss
                num_losses += 1
                batch_inputs.clear()
                batch_labels.clear()

#                if random.random() <= 0.02:
#                    print('TRAIN image ', image_path)
#                    print('Output ', outputs, ' labels ', labels)
#    #                print('Output ', torch.argmax(outputs), ' labels ', labels)
#                    print('Loss = ', loss)
#                    print()
        val_loss = 0.0
        # Validation testing
        with torch.no_grad():
            for image_path in img_to_val_labels:
                labels = img_to_val_labels[image_path]
                ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
                x = torch.from_numpy(framed_imgs[0])
                if use_cuda:

                    labels = labels.cuda(gpu)
                    x = x.cuda(gpu)
                    if use_float16:
                        x = x.half()
                    else:
                        x = x.float()
                else:
                    x = x.float()

                x = x.unsqueeze(0).permute(0, 3, 1, 2)
                outputs = model(x)


                loss = criterion(outputs, labels)
                val_loss += loss
                if random.random() <= 0.03:
                    print('VAL image ', image_path)
                    print('Output ', outputs, ' labels ', labels)
#                    print('Output ', torch.argmax(outputs), ' labels ', labels)
                    print('Loss = ', loss)
                    print()
        if epoch > 4 and val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(saved_path, f'efficientnet-d0-counter_min.pth'))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, f'efficientnet-d0-counter_{epoch}.pth'))
        avg_loss = total_loss.item() / num_losses
        all_train_losses.append(avg_loss)
        all_val_losses.append(val_loss.item() / len(img_to_val_labels))
        print('total loss ', avg_loss)
        print('val loss ', val_loss.item() / len(img_to_val_labels))
        if epoch % 5 == 0:
            print('All train losses ', all_train_losses)
            print('All val losses ', all_val_losses)


if __name__ == '__main__':

    image_paths = []
    
    model = EfficientNetCounter(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))

    if args.weights is not None:
        print('Loading reg weights')
        model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    else:       
        print('Loading backbone')
        model.backbone_net.load_state_dict(torch.load(backbone_weights, map_location=torch.device('cpu')))
        model.bifpn.load_state_dict(torch.load(bifpn_weights, map_location=torch.device('cpu')))
#    for param in model.backbone_net.parameters():
#        param.requires_grad = False


    print('loaded weights ')
    
    model.eval()
    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()

    train(input_file, val_file, model, saved_path, params)

