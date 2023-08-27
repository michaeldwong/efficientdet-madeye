

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os
import train
import argparse
import torch
import yaml
import random
import shutil

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
import pandas as pd

ap = argparse.ArgumentParser()

ap.add_argument('inference', type=str, help='Inference dir')
ap.add_argument('rectlinear', type=str, help='Rectlinear dir')

ap.add_argument('first_frame', type=int, help='First frame in the frame boundary')
ap.add_argument('frame_begin', type=int, help='Begin frame to start evaluation from')
ap.add_argument('frame_limit', type=int, help='End frame')


ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')


ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=1)
ap.add_argument('--float16', type=boolean_string, default=False)
args = ap.parse_args()



PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0
compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
project_name = args.project
weights_path = f'{project_name}/{project_name}/efficientdet-d{compound_coef}_val_min.pth' if args.weights is None else args.weights

data_path = 'continual-learning-temp/datasets/'
saved_path = 'continual-learning-temp/weights/'


print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = train.Params(f'projects/{project_name}.yml')


obj_list = params.obj_list

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]







class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss







def generate_neighboring_orientations(current_orientation):
    items = current_orientation.split('-')
    pan = int(items[0])
    zoom = int(items[-1])
    if pan == 0:
        left_horz = 330
    else:
        left_horz = int(items[0]) - 30
    if pan == 330:
        right_horz = 0
    else:
        right_horz = int(items[0]) + 30

    if len(items) == 4:
        tilt = int(items[2]) * -1
    else:
        tilt = int(items[1])
    top_tilt = tilt + 15
    bottom_tilt = tilt - 15

    if tilt == 30:
        return [ f'{left_horz}-{tilt}-{zoom}', 
                 f'{right_horz}-{tilt}-{zoom}', 
                 current_orientation, 
                 f'{right_horz}-{bottom_tilt}-{zoom}', 
                 f'{pan}-{bottom_tilt}-{zoom}',
                 f'{left_horz}-{bottom_tilt}-{zoom}',
                ]
    elif tilt == -30:
        return [ 
                 f'{left_horz}-{tilt}-{zoom}', 
                 f'{right_horz}-{tilt}-{zoom}', 
                 current_orientation, 
                 f'{right_horz}-{top_tilt}-{zoom}', 
                 f'{pan}-{top_tilt}-{zoom}',
                 f'{left_horz}-{top_tilt}-{zoom}',
                ]

    return [ 
             f'{left_horz}-{top_tilt}-{zoom}',
             f'{left_horz}-{tilt}-{zoom}', 
             f'{left_horz}-{bottom_tilt}-{zoom}', 
             f'{right_horz}-{top_tilt}-{zoom}', 
             f'{right_horz}-{tilt}-{zoom}', 
             f'{right_horz}-{bottom_tilt}-{zoom}', 
             current_orientation, 
             f'{pan}-{top_tilt}-{zoom}', 
             f'{pan}-{bottom_tilt}-{zoom}' ]








def create_annotations(f, image_file, orientation_df, orientation, object_type, json_dict, image_id, annotation_id):
    if object_type != 'car' and object_type != 'person' and object_type != 'both':
        raise Exception('Incorrect object type')
    count = 0
    for idx, row in orientation_df.iterrows():
        if object_type == 'car' or object_type == 'both':
            if row['class'] == 'car' and row['confidence'] >= CAR_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                json_dict['annotations'].append({"id": annotation_id,"image_id": image_id, "category_id": 1, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin ], "area": (ymax - ymin) * (xmax - xmin), "segmentation": [[xmin, ymin, xmax , ymin, xmax, ymax, xmin, ymax ]] })
        if object_type == 'person' or object_type == 'both':
            if row['class'] == 'person' and row['confidence'] >= PERSON_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                json_dict['annotations'].append({"id": annotation_id, "image_id": image_id, "category_id": 2, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin ], "area": (ymax - ymin) * (xmax - xmin), "segmentation": [[xmin, ymin, xmax , ymin, xmax, ymax, xmin, ymax ]] })
        annotation_id += 1
    return annotation_id

# set_type is 'train' or 'val'
def generate_dataset(inference_dir, rectlinear_dir,  current_frame, orientation_to_frames, set_type, data_path,  project_name):
    json_dict = {
        "info": {
            "description": "","url": "","version": "1.0","year": 2017,"contributor": "","date_created": "2017/09/01"
        },
        "licenses": [
                {"id": 1, "name": "None", "url": "None"}
        ],
        "images": [

        ],
        "annotations": [

        ],
        "categories": [
            {"id": 1, "name": "car", "supercategory": "None"},
            {"id": 2, "name": "person", "supercategory": "None"},
        ],
    }

    image_id = 0
    annotation_id = 0
    image_outdir = f'{data_path}/{project_name}/{set_type}'
#    if os.path.exists(image_outdir):
#        shutil.rmtree(image_outdir)
    annotations_outdir = f'{data_path}/{project_name}/annotations/'
#    if os.path.exists(os.path.join(annotations_outdir, f'instances_{set_type}.json')):
#        shutil.rmtree(os.path.join(annotations_outdir, f'instances_{set_type}.json'))
    os.makedirs(annotations_outdir, exist_ok=True)
    os.makedirs(image_outdir, exist_ok=True)

    print('generating dataset')
    print(orientation_to_frames)
    for o in orientation_to_frames:
        frames = orientation_to_frames[o]
        result_orientation_dir = os.path.join(inference_dir,  o)
        for f in frames:
            inference_file = os.path.join(result_orientation_dir, f'frame{f}.csv')
            if os.path.getsize(inference_file) > 0:
                orientation_df = pd.read_csv(inference_file)
                orientation_df.columns = ['left', 'top', 'right', 'bottom', 'class', 'confidence']
                orig_image_file = os.path.join(rectlinear_dir, o, f'frame{current_frame}.jpg')
                image_file = f'{o}-frame{current_frame}.jpg'
                dest = f'{image_outdir}/{image_file}'
                shutil.copy(orig_image_file, dest)
                json_dict['images'].append({"id": image_id, "file_name": image_file, "width": 1280, "height": 720, "date_captured": "", "license": 1, "coco_url": "", "flickr_url": ""})
                annotation_id = create_annotations(f, image_file, orientation_df, o, 'both', json_dict, image_id, annotation_id)
            image_id += 1

    with open(os.path.join(annotations_outdir , f'instances_{set_type}.json'), 'w') as f_out:
        json.dump(json_dict, f_out)

def save_checkpoint_continual_learning(model, name, saved_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))

def continual_train(  weights_path, data_path, saved_path, project_name, gpu, num_epochs=10):

    use_cuda = gpu >= 0
    # Hardcoded options
    batch_size = 4
    compound_coef = 0
    # Input to dataset
    head_only = True
    val_interval = 1
    es_min_delta = 0.0
    num_workers = 12
    lr = 1e-4
    optim = 'adamw'
    es_patience = 0
    save_interval = 100
  
    best_weights = weights_path
 
    os.makedirs(os.path.join(data_path, project_name ), exist_ok=True)
    os.makedirs(os.path.join(saved_path, project_name ), exist_ok=True)


    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    if gpu >= 0:
        torch.cuda.set_device(f'cuda:{gpu}')



    training_params = {'batch_size': batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': num_workers}

    val_params = {'batch_size': batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    print('Reading from ', os.path.join(data_path, project_name))
    print(params)
    
    print('train set ', params.train_set)
    training_set = CocoDataset(root_dir=os.path.join(data_path, project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(data_path, project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    try:
        ret = model.load_state_dict(torch.load(weights_path  ),strict=False)
        if gpu >= 0:
            model.to(f'cuda:{gpu}')
    except RuntimeError as e:
        print(f'[Warning] Ignoring {e}')
        print(
            '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

    print(f'[Info] loaded weights: {os.path.basename(weights_path)}')
    # freeze backbone if train head_only
    if head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    model.train()

    step = 0
    num_iter_per_epoch = len(training_generator)
    min_loss = 0.0
    try:
        for epoch in range(num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    if params.num_gpus == 1 and gpu >= 0:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
#                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % save_interval == 0 and step > 0:
                        save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name))
                    if loss < min_loss:
                        save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_min.pth', os.path.join(saved_path, project_name ))

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            # VAlidation testing
#            if epoch % val_interval == 0:
#                model.eval()
#                loss_regression_ls = []
#                loss_classification_ls = []
#                for iter, data in enumerate(val_generator):
#                    with torch.no_grad():
#                        imgs = data['img']
#                        annot = data['annot']
#
#                        if params.num_gpus == 1:
#                            imgs = imgs.cuda()
#                            annot = annot.cuda()
#
#                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
#                        cls_loss = cls_loss.mean()
#                        reg_loss = reg_loss.mean()
#
#                        loss = cls_loss + reg_loss
#                        if loss == 0 or not torch.isfinite(loss):
#                            continue
#
#                        loss_classification_ls.append(cls_loss.item())
#                        loss_regression_ls.append(reg_loss.item())
#
#                cls_loss = np.mean(loss_classification_ls)
#                reg_loss = np.mean(loss_regression_ls)
#                loss = cls_loss + reg_loss
#
#                print(
#                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
#                        epoch, num_epochs, cls_loss, reg_loss, loss))
#
#                if loss + es_min_delta < best_loss:
#                    best_loss = loss
#                    best_epoch = epoch
#
#                    save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name))
#                    best_weights = f'continual-learning/weights/efficientdet-d{compound_coef}_min.pth'
#                model.train()
#
#                # Early stopping
#                if epoch - best_epoch > es_patience > 0:
#                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
#
#                    break
    except KeyboardInterrupt:
        save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name))

    save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name))
    if len(os.listdir(os.path.join(saved_path, project_name))) == 0:
        print('NO WEIGHTS SAVED')
        best_weights = weights_path
    else: 
        best_weights = get_last_weights(os.path.join(saved_path, project_name))
    print('best weights ', best_weights)
    return best_weights




def rank_orientations(orientation_to_count):
    sorted_dict = {k: v for k, v in sorted(orientation_to_count.items(), key=lambda item: item[1] * -1)}
    orientation_to_rank = {}
    last_count = 0
    rank = 0
    for o in sorted_dict:
        count = sorted_dict[o]
        if count != last_count:
            last_count = count
            rank += 1
        if rank == 0:
            rank += 1
        orientation_to_rank[o] = rank
    return orientation_to_rank

def evaluate_orientation_with_df(orientation_df):
    car_count = 0
    person_count = 0
    for idx, row in orientation_df.iterrows():
        if row['class'] == 'car' and row['confidence'] >= 70.0:
            coords = [row['top'], row['bottom'], row['left'], row['right']]
            car_count += 1
        if row['class'] == 'person' and row['confidence'] >= 50.0:
            coords = [row['top'], row['bottom'], row['left'], row['right']]
            person_count += 1
    return car_count, person_count


#def get_counts(frame, orientation):
#    pd.read_csv(f'/home/mikew/inference-results/seattle-dt-2-small/seattle-dt-2-6fps-imgsize1/yolov4/{orientation}/frame{frame}.csv')
#    car_count = 0
#    person_count = 0
#    for idx, row in orientation_df.iterrows():
#        if row['class'] == 'car' and row['confidence'] >= 70.0:
#            coords = [row['top'], row['bottom'], row['left'], row['right']]
#            car_count += 1
#        if row['class'] == 'person' and row['confidence'] >= 50.0:
#            coords = [row['top'], row['bottom'], row['left'], row['right']]
#            person_count += 1
#    return car_count,person_count

def generate_orientations():
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    orientations = []
    for r1 in range(0,360,30):
#        for r2 in  [ -60, -30, -15, 0, 15, 30, 60]:
        for r2 in  [ -30, -15, 0, 15, 30]:
            for z in [1, 2]:
                orientations.append(f'{r1}-{r2}-{z}')
    return orientations

def extract_rotation(o):
    return o.split('-')[0]

def best_fixed_orientations(current_frame, frame_limit, inference, orientations):
    orientation_to_car_count = {}
    orientation_to_person_count = {}
    begin_frame = current_frame
    for o in orientations:
        orientation_to_car_count[o] = 0
        orientation_to_person_count[o] = 0
        current_frame = begin_frame
        while current_frame <= frame_limit:
            if current_frame % 6 != 0:
                current_frame += 1
                continue
            inference_file = os.path.join(inference, o, f'frame{current_frame}.csv')
            current_frame += 1

            if  os.path.getsize(inference_file) == 0:
                continue
            car_count, person_count = evaluate_orientation_with_df(pd.read_csv(inference_file))
            orientation_to_car_count[o] += car_count
            orientation_to_person_count[o] += person_count
    # Car count
    highest_count = 0
    best_car_orientation = '0-0-1'
    for o in orientation_to_car_count:
        if orientation_to_car_count[o] > highest_count:
            best_car_orientation = o
            highest_count = orientation_to_car_count[o] 
    highest_count = -1
#    current_rotation = extract_rotation(best_car_orientation)
#    next_best_car_orientation = '0-0-1'
#    orientations_to_exclude = generate_neighboring_orientations(best_car_orientation)
#
#    for o in orientation_to_car_count:
#        if o in orientations_to_exclude or extract_rotation(o) == current_rotation:
#            continue
#        new_neighbors = generate_neighboring_orientations(o)
#        for no in new_neighbors:
#            if no in orientations_to_exclude:
#                continue
#        if orientation_to_car_count[o] > highest_count:
#            next_best_car_orientation = o
#            highest_count = orientation_to_car_count[o]

    # Ppl count
    highest_count = 0
    best_person_orientation = '0-0-1'

    current_rotation = extract_rotation(best_car_orientation)
    orientations_to_exclude = generate_neighboring_orientations(best_car_orientation)
    for o in orientation_to_person_count:

        if o in orientations_to_exclude or extract_rotation(o) == current_rotation:
            continue
        if orientation_to_person_count[o] > highest_count:

            best_person_orientation = o
            highest_count = orientation_to_person_count[o] 


#    highest_count = -1
#    current_rotation = extract_rotation(best_person_orientation)
#    next_best_person_orientation = '0-0-1'
#    orientations_to_exclude = generate_neighboring_orientations(best_person_orientation)
#
#    for o in orientation_to_person_count:
#        if o in orientations_to_exclude or extract_rotation(o) == current_rotation:
#            continue
#        new_neighbors = generate_neighboring_orientations(o)
#        for no in new_neighbors:
#            if no in orientations_to_exclude:
#                continue
#        if orientation_to_person_count[o] > highest_count:
#            next_best_person_orientation = o
#            highest_count = orientation_to_person_count[o]
#
#    return [best_car_orientation, best_person_orientation, next_best_car_orientation, next_best_person_orientation] 
    return [best_car_orientation, best_person_orientation]

def evaluate_coco(  model, threshold=0.05):

    gt_orientation_to_car_selected = {}
    gt_orientation_to_person_selected = {}
    orientation_to_person_selected = {}
    orientation_to_car_selected = {}

    base_orientations = best_fixed_orientations(args.first_frame, args.frame_limit, args.inference, generate_orientations())
    orientation_to_training_frames = {}
    orientation_to_val_frames = {}
    print('base orientations ', base_orientations)
    for base_idx,current_orientation  in enumerate(base_orientations):

        neighboring_orientations = generate_neighboring_orientations(current_orientation)
        
        current_frame = args.frame_begin

        best_fixed_car_orientation = current_orientation
        best_fixed_person_orientation = current_orientation
        best_fixed_both_orientation = current_orientation

        best_fixed_car_count = 0
        best_fixed_person_count = 0
        print('Current orientation ', current_orientation)



        orientation_to_agg_car_rank = {}
        orientation_to_agg_person_rank = {}
        orientation_to_agg_both_rank = {}
        while current_frame <= args.frame_limit:
            if current_frame % 6 != 0:
                current_frame += 1
                continue

            gt_orientation_to_car_count = {}
            gt_orientation_to_person_count = {}

            max_person_count = 0
            max_car_count = 0
            for no in neighboring_orientations:
                inference_file = os.path.join(args.inference, no, f'frame{current_frame}.csv')
                if not os.path.exists(inference_file):
                    current_frame += 1
                    print(inference_file, ' DNE')
                    continue
                elif  os.path.getsize(inference_file) == 0:
                    current_frame += 1
                    print(inference_file , ' size is 0')
                    continue

                gt_car_count, gt_person_count = evaluate_orientation_with_df(pd.read_csv(inference_file))
                gt_orientation_to_car_count[no] = gt_car_count
                gt_orientation_to_person_count[no] = gt_person_count
                if gt_car_count > max_car_count:
                    max_car_count = gt_car_count
                if gt_person_count > max_person_count:
                    max_person_count = gt_person_count

            if max_car_count == 0:
                max_car_count = 1
            if max_person_count == 0:
                max_person_count = 1
            gt_orientation_to_both_count = {}
            for o in gt_orientation_to_car_count:
                gt_orientation_to_both_count[o] = (gt_orientation_to_person_count[o] / max_person_count) +( gt_orientation_to_car_count[o] / max_car_count)

            gt_orientation_to_car_rank = rank_orientations(gt_orientation_to_car_count)
            gt_orientation_to_person_rank = rank_orientations(gt_orientation_to_person_count)
            gt_orientation_to_both_rank = rank_orientations(gt_orientation_to_both_count)

            for no in neighboring_orientations:
                if no not in orientation_to_agg_car_rank:
                    orientation_to_agg_car_rank[no] = []
                orientation_to_agg_car_rank[no].append(gt_orientation_to_car_rank[no])

                if no not in orientation_to_agg_person_rank:
                    orientation_to_agg_person_rank[no] = []
                orientation_to_agg_person_rank[no].append(gt_orientation_to_person_rank[no])

                if no not in orientation_to_agg_both_rank:
                    orientation_to_agg_both_rank[no] = []
                orientation_to_agg_both_rank[no].append(gt_orientation_to_both_rank[no])

            current_frame += 1

        best_avg_car_rank = 10.0
        best_avg_person_rank = 10.0
        best_avg_both_rank = 10.0
        for no in neighboring_orientations:

            both_rank = sum(orientation_to_agg_both_rank[no]) / len(orientation_to_agg_both_rank[no])
            car_rank = sum(orientation_to_agg_car_rank[no]) / len(orientation_to_agg_car_rank[no])
            person_rank = sum(orientation_to_agg_person_rank[no]) / len(orientation_to_agg_person_rank[no])
            if car_rank < best_avg_car_rank:
                best_avg_car_rank = car_rank
                best_fixed_car_orientation = no
            if person_rank < best_avg_person_rank:
                best_avg_person_rank = person_rank
                best_fixed_person_orientation = no
            if both_rank < best_avg_both_rank:
                best_avg_both_rank = both_rank
                best_fixed_both_orientation = no

        # Get best fixed info
#        for no in neighboring_orientations:
#            running_car_count = 0
#            running_person_count = 0
#            while current_frame <= args.frame_limit:
#                if current_frame % 6 != 0:
#                    current_frame += 1
#                    continue
#                inference_file = os.path.join(args.inference, no, f'frame{current_frame}.csv')
#                if not os.path.exists(inference_file):
#                    current_frame += 1
#                    print(inference_file, ' DNE')
#                    continue
#                elif  os.path.getsize(inference_file) == 0:
#                    current_frame += 1
#                    print(inference_file , ' size is 0')
#                    continue
#
#                gt_car_count, gt_person_count = evaluate_orientation_with_df(pd.read_csv(inference_file))
#                running_car_count += gt_car_count
#                running_person_count += gt_person_count
#                current_frame += 1
#            if running_car_count > best_fixed_car_count:
#                best_fixed_car_orientation = no
#                best_fixed_car_count = running_car_count
#
#            if running_person_count > best_fixed_person_count:
#                best_fixed_person_orientation = no
#                best_fixed_person_count = running_person_count
        print('Got best fixed information')

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        frame_to_orientation_to_car_count = {}
        frame_to_orientation_to_person_count = {}

        gt_orientation_to_car_count = {}
        gt_orientation_to_person_count = {}
        gt_orientation_to_both_count = {}

        overall_car_ranks = []
        overall_person_ranks = [] 

        overall_fixed_car_ranks = []
        overall_fixed_person_ranks = []


        overall_both_ranks = []
        overall_fixed_both_ranks = []

        current_frame = args.frame_begin

        print('Beginning actual evaluation')
        while current_frame <= args.frame_limit:
            if current_frame % 6 != 0:
                current_frame += 1
                continue
            print()
            print('Frame ', current_frame)
            orientation_to_car_count = {}

            print('Num neighbors ', len(neighboring_orientations))
            thresh_to_orientation_to_car_count = {}
            thresh_to_orientation_to_both_count = {}
            thresh_to_orientation_to_person_count = {}

            max_gt_car_count = 0
            max_gt_person_count = 0
            for no in neighboring_orientations:

                inference_file = os.path.join(args.inference, no, f'frame{current_frame}.csv')
                if not os.path.exists(inference_file):
                    print(inference_file, ' DNE')
                    continue
                elif  os.path.getsize(inference_file) == 0:
                    print(inference_file , ' size is 0')
                    continue
                gt_car_count, gt_person_count = evaluate_orientation_with_df(pd.read_csv(inference_file))
                gt_orientation_to_car_count[no] = gt_car_count
                gt_orientation_to_person_count[no] = gt_person_count
                if gt_car_count > max_gt_car_count:
                    max_gt_car_count = gt_car_count
                if gt_person_count > max_gt_person_count:
                    max_gt_person_count = gt_person_count
                image_path = os.path.join(args.rectlinear, no, f'frame{current_frame}.jpg')

                ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params.mean, std=params.std)
                x = torch.from_numpy(framed_imgs[0])

                if use_cuda:
                    x = x.cuda(gpu)
                    if use_float16:
                        x = x.half()
                    else:
                        x = x.float()
                else:
                    x = x.float()

                person_count = 0
                car_count = 0
                x = x.unsqueeze(0).permute(0, 3, 1, 2)
                features, regression, classification, anchors = model(x)

                preds = postprocess(x,
                                    anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    threshold, nms_threshold)
                if not preds:
                    print('NOT preds for orientation ', no, ' frame ', current_frame)
                    continue
                preds = invert_affine(framed_metas, preds)[0]

                scores = preds['scores']
                class_ids = preds['class_ids']
                rois = preds['rois']
                for thresh in [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
                    person_count = 0
                    car_count = 0
                    for i in range(0,len(scores)):
                        # For COCO Class id 2 is car, 0 is person
        #                        if scores[i] >= 0.3  and class_ids[i] == 2 :
        #                            car_count += 1
        #                        elif scores[i] >= 0.2 and class_ids[i] == 0:
        #                            people_count += 1
                        # Class id 0 is car, 1 is person
                        if scores[i] >= thresh  and class_ids[i] == 0 :
                            car_count += 1
                        elif scores[i] >= thresh and class_ids[i] == 1:
                            person_count += 1
                    if thresh not in thresh_to_orientation_to_car_count:
                        thresh_to_orientation_to_car_count[thresh] = {}
                    thresh_to_orientation_to_car_count[thresh][no] = car_count
                    if thresh not in thresh_to_orientation_to_person_count:
                        thresh_to_orientation_to_person_count[thresh] = {}
                    thresh_to_orientation_to_person_count[thresh][no] = person_count

            if max_gt_car_count == 0:
                max_gt_car_count = 1
            if max_gt_person_count == 0:
                max_gt_person_count = 1
            for o in neighboring_orientations:

                gt_orientation_to_car_count[o] /= max_gt_car_count
                gt_orientation_to_person_count[o] /= max_gt_person_count
                gt_orientation_to_both_count[o] = gt_orientation_to_car_count[o] + gt_orientation_to_person_count[o]

            current_frame += 1


            gt_orientation_to_car_rank = rank_orientations(gt_orientation_to_car_count)
            gt_orientation_to_person_rank = rank_orientations(gt_orientation_to_person_count)
            gt_orientation_to_both_rank = rank_orientations(gt_orientation_to_both_count)

            best_car_thresh = 0.3
            best_person_thresh = 0.1
            best_car_rank = 10
            best_person_rank = 10
            for thresh in [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
                tmp_orientation_to_car_count = thresh_to_orientation_to_car_count[thresh]
                tmp_orientation_to_person_count = thresh_to_orientation_to_person_count[thresh]
                tmp_orientation_to_car_rank = rank_orientations(tmp_orientation_to_car_count)
                tmp_orientation_to_person_rank = rank_orientations(tmp_orientation_to_person_count)
                best_car_orientation  = neighboring_orientations[0]
                best_person_orientation  = neighboring_orientations[0]
                for no in neighboring_orientations:
                    if tmp_orientation_to_car_rank[no] == 1 and gt_orientation_to_car_rank[no] < best_car_rank:
                        best_car_thresh = thresh
                        best_car_rank = gt_orientation_to_car_rank[no]
                    elif tmp_orientation_to_person_rank[no] == 1 and gt_orientation_to_person_rank[no] < best_person_rank:
                        best_person_thresh = thresh
                        best_person_rank = gt_orientation_to_person_rank[no]


            orientation_to_car_count = thresh_to_orientation_to_car_count[best_car_thresh]
            orientation_to_person_count = thresh_to_orientation_to_person_count[best_person_thresh]
            print('car thresh = ', best_car_thresh)
            print('person thresh ', best_person_thresh)
            max_car_count = 0
            max_person_count = 0
            for no in neighboring_orientations:
                car_count = orientation_to_car_count[no]
                person_count = orientation_to_person_count[no]
                if car_count > max_car_count:
                    max_car_count = car_count
                if person_count > max_person_count:
                    max_person_count = person_count
                print(no , ' : cars -> ', car_count, ' ppl  -> ', person_count)
            if max_car_count == 0:
                max_car_count = 1
            if max_person_count == 0:
                max_person_count = 1

            orientation_to_both_count = {}
            for o in neighboring_orientations:
                orientation_to_both_count[o] = (orientation_to_car_count[o] / max_car_count) + (orientation_to_person_count[o] / max_person_count)

            orientation_to_car_rank = rank_orientations(orientation_to_car_count)
            orientation_to_person_rank = rank_orientations(orientation_to_person_count)
            orientation_to_both_rank = rank_orientations(orientation_to_both_count)












            correct = 0
            car_ranks = []
            person_ranks = []
            both_ranks = []

            top_car_orientations = []
            top_person_orientations = []
            top_both_orientations = []

            max_car_count = 0

            max_person_count = 0
            max_both_count = 0


            for o in orientation_to_car_count:
                if orientation_to_car_count[o] > max_car_count:
                    max_car_count = orientation_to_car_count[o]
            if max_car_count == 0:
                car_ranks.append(gt_orientation_to_car_rank[best_fixed_car_orientation])
                top_car_orientations.append(best_fixed_car_orientation)
            else:
                for o in orientation_to_car_rank:

                    if o not in orientation_to_car_selected:
                        orientation_to_car_selected[o] = 0
                    if orientation_to_car_rank[o] == 1:
                        car_ranks.append(gt_orientation_to_car_rank[o])
                        top_car_orientations.append(o)
                        orientation_to_car_selected[o] += 1

                        if o not in orientation_to_training_frames:
                            orientation_to_training_frames[o] = []
                        if o not in orientation_to_val_frames:
                            orientation_to_val_frames[o] = []
                        if random.random() <= 0.8:
                            orientation_to_training_frames[o].append(current_frame)
                        else:
                            orientation_to_val_frames[o].append(current_frame)

                for o in gt_orientation_to_car_rank:
                    if o not in gt_orientation_to_car_selected:
                        gt_orientation_to_car_selected[o] = 0
                    if gt_orientation_to_car_rank[o] == 1:
                        gt_orientation_to_car_selected[o] += 1


#            current_car_rank = sum(car_ranks) / len(car_ranks)
            current_car_rank = min(car_ranks)


            for o in orientation_to_person_count:
                if orientation_to_person_count[o] > max_person_count:
                    max_person_count = orientation_to_person_count[o]
            if max_person_count == 0:
                person_ranks.append(gt_orientation_to_person_rank[best_fixed_person_orientation])
                top_person_orientations.append(best_fixed_person_orientation)
            else:
                for o in orientation_to_person_rank:
                    if orientation_to_person_rank[o] == 1:
                        if o not in orientation_to_person_selected:
                            orientation_to_person_selected[o] = 0
                        person_ranks.append(gt_orientation_to_person_rank[o])
                        top_person_orientations.append(o)
                        orientation_to_person_selected[o] += 1

                        if o not in orientation_to_training_frames:
                            orientation_to_training_frames[o] = []
                        if o not in orientation_to_val_frames:
                            orientation_to_val_frames[o] = []
                        if random.random() <= 0.8:
                            orientation_to_training_frames[o].append(current_frame)
                        else: 
                            orientation_to_val_frames[o].append(current_frame)

                for o in gt_orientation_to_person_rank:
                    if o not in gt_orientation_to_person_selected:
                        gt_orientation_to_person_selected[o] = 0
                    if gt_orientation_to_person_rank[o] == 1:
                        gt_orientation_to_person_selected[o] += 1

            print('Current orientation ', current_orientation )
            print('Top car orientations ', top_car_orientations)
            print('Top person orienttions ', top_person_orientations)
            if max_gt_car_count > 0:
                fixed_car_rank = gt_orientation_to_car_rank[best_fixed_car_orientation]
                overall_fixed_car_ranks.append(fixed_car_rank)
                print('Car counts ', orientation_to_car_count)
                print('GT Car counts ', gt_orientation_to_car_count)
                print('Fixed car rank ', fixed_car_rank)
                print('Car rank ', current_car_rank)
                overall_car_ranks.append(current_car_rank)
#            current_person_rank = sum(person_ranks) / len(person_ranks)
            current_person_rank = min(person_ranks)
            if max_gt_person_count > 0:
                fixed_person_rank = gt_orientation_to_person_rank[best_fixed_person_orientation]
                overall_fixed_person_ranks.append(fixed_person_rank)
                print('Person counts ', orientation_to_person_count)
                print('GT person counts ', gt_orientation_to_person_count)
                print('Fixed person rank ', fixed_person_rank)
                print('Person rank ', current_person_rank)
                overall_person_ranks.append(current_person_rank)


            for o in orientation_to_both_count:
                if orientation_to_both_count[o] > max_both_count:
                    max_both_count = orientation_to_both_count[o]
            if max_both_count == 0:
                both_ranks.append(gt_orientation_to_both_rank[best_fixed_both_orientation])
                top_both_orientations.append(best_fixed_both_orientation)
            else:
                for o in orientation_to_both_rank:
                    if orientation_to_both_rank[o] == 1:
                        both_ranks.append(gt_orientation_to_both_rank[o])
                        top_both_orientations.append(o)

            current_both_rank = sum(both_ranks) / len(both_ranks)
            if max_both_count > 0:
                fixed_both_rank = gt_orientation_to_both_rank[best_fixed_both_orientation]
                overall_fixed_both_ranks.append(fixed_both_rank)
                print('Both counts ', orientation_to_both_count)
                print('GT both counts ', gt_orientation_to_both_count)
                print('Fixed both rank ', fixed_both_rank)
                print('Both rank ', current_both_rank)
                overall_both_ranks.append(current_both_rank)


            # Continual learning
            if current_frame % 360 == 0:
                print('Starting continual learning')
                print('Current training frames (no augmentation) ', orientation_to_training_frames)
                print('Current val frames (no augmentation) ', orientation_to_val_frames)
                temp_project_name = 'test'
                if os.path.exists(f'{saved_path}/{temp_project_name}'):
                    shutil.rmtree(f'{saved_path}/{temp_project_name}')
                if os.path.exists(f'{data_path}/{temp_project_name}'):
                    shutil.rmtree(f'{data_path}/{temp_project_name}')

                # Augment with historical frames
                largest_train_frame_count = 0
                largest_val_frame_count = 0
                for o in orientation_to_training_frames:
                    if len(orientation_to_training_frames[o]) > largest_train_frame_count:
                        largest_train_frame_count = len(orientation_to_training_frames[o])
                    if len(orientation_to_val_frames[o]) > largest_val_frame_count:
                        largest_val_frame_count = len(orientation_to_val_frames[o])

                for f_num in range(args.frame_begin + 720, args.frame_begin, -1):
                    if f_num % 6 != 0:
                        continue
                    if random.random() <= 0.4:
                        for no in neighboring_orientations:
                            if no not in orientation_to_training_frames:
                                orientation_to_training_frames[no] = []
                            if len(orientation_to_training_frames[no]) < largest_train_frame_count:
                                orientation_to_training_frames[no].append(f_num)
                    else:
                        for no in neighboring_orientations:
                            if no not in orientation_to_val_frames:
                                orientation_to_val_frames[no] = []
                            if len(orientation_to_val_frames[no]) < largest_val_frame_count:
                                orientation_to_val_frames[no].append(f_num)

                # Train set
                generate_dataset(args.inference, args.rectlinear,  current_frame, orientation_to_training_frames, 'train', data_path, temp_project_name)
        #        # Val set
                generate_dataset(args.inference, args.rectlinear,  current_frame, orientation_to_val_frames, 'val', data_path,  temp_project_name)
                new_weights_path = continual_train( weights_path, data_path, saved_path, 'test', gpu, num_epochs=5)
                print('Saved weights ', new_weights_path)
                model.load_state_dict(torch.load(new_weights_path))
                model.requires_grad_(False)
                orientation_to_val_frames.clear()
                orientation_to_training_frames.clear()
            current_frame += 1


        if len(overall_car_ranks) == 0:
            car_rank = -1.0
        else:
            car_rank = sum(overall_car_ranks) / len(overall_car_ranks) 
            fixed_car_rank = sum(overall_fixed_car_ranks) / len(overall_fixed_car_ranks)
            
        if len(overall_person_ranks) == 0:
            person_rank = -1.0
        else:
            person_rank = sum(overall_person_ranks) / len(overall_person_ranks) 
            fixed_person_rank = sum(overall_fixed_person_ranks) / len(overall_fixed_person_ranks)
        if len(overall_both_ranks) == 0:
            both_rank = -1.0
        else:
            both_rank  = sum(overall_both_ranks) / len(overall_both_ranks)
            fixed_both_rank  = sum(overall_fixed_both_ranks) / len(overall_fixed_both_ranks)

        if base_idx == 0:
            print('BEST CAR ORIENTATION')
        else:
            print('BEST PERSON ORIENTATION')
        print('****')
        print('base orientation ', current_orientation)

        print('Overall fixed Car rank is ', fixed_car_rank)
        print('Overall Car rank is ', car_rank)
        print('Overall fixed person rank is ', fixed_person_rank)
        print('Overall person rank is ', person_rank)
        print('Overall fixed both rank is ', fixed_both_rank)
        print('Overall both rank is ', both_rank)
        print()
        print('Car selection distribution ', orientation_to_car_selected)
        print('ground truth Car selection distribution ', orientation_to_car_selected)
        print('Person selection distribution ', orientation_to_person_selected)
        print('ground truth person selection distribution ', orientation_to_person_selected)


if __name__ == '__main__':

    image_paths = []

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    model.load_state_dict(torch.load(weights_path))
    model.requires_grad_(False)
    model.eval()

    current_net = model.backbone_net
    prev_net = model.backbone_net

    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()
    print('Starting ... ')
    evaluate_coco(model)

