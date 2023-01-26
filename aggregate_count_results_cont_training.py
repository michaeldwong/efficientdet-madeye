

import pandas as pd
import numpy as np 
import cv2
import pickle
import random
import os
import argparse
import shutil
import time

import json
import os

import argparse
import torch
import yaml

import train
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string


CAR_CONFIDENCE_THRESH = 70.0
PERSON_CONFIDENCE_THRESH = 50.0
# For the params we care about, hardcoding to this yml file is okay

SKIP = 6

nms_threshold = 0.5
params = train.Params(f'projects/madeye.yml')
obj_list = params.obj_list
compound_coef = 0
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]




use_float16 = False
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))




def extract_csv_results(infile, car_thresh, person_thresh):
    car_objects = []
    person_objects = []
    df = pd.read_csv(infile, names=['left', 'top', 'right', 'bottom', 'class', 'confidence'])
    for idx, row in df.iterrows():
        if row['class'] == 'car':
            if float(row['confidence']) >= car_thresh:
                car_objects.append([float(row['left']), float(row['top']), float(row['right']), float(row['bottom'])])
        if row['class'] == 'person':
            if float(row['confidence']) >= person_thresh:
                person_objects.append([float(row['left']), float(row['top']), float(row['right']), float(row['bottom'])])
    return car_objects, person_objects


def run_inference(orientation_to_file,  model, gpu, threshold=0.05):

    use_cuda = gpu >= 0
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    orientation_to_car_count = {}
    orientation_to_person_count = {}
    # In format frame,orientation,file
    with torch.no_grad():
        for o in orientation_to_file:
            image_path  = orientation_to_file[o]
            print('Processing ', image_path)
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

            people_count = 0
            car_count = 0
            x = x.unsqueeze(0).permute(0, 3, 1, 2)
            features, regression, classification, anchors = model(x)

            preds = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, nms_threshold)
            if not preds:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                f_results.write(f'{frame},{orientation},{car_count},{people_count}\n')
                continue

            preds = invert_affine(framed_metas, preds)[0]

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']
    #        print('Scores ', scores)
    #        print("Class ids ", class_ids)
    #        print('ROIs ', rois)
            idx = 0
            if len(scores) == 0:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                continue
            for i in range(0,len(scores)):
                # Class id 0 is car, 1 is person
                if scores[i] >= 0.7  and class_ids[i] == 0 :
                    car_count += 1
                elif scores[i] >= 0.5 and class_ids[i] == 1:
                    people_count += 1
            orientation_to_car_count[o] = car_count
            orientation_to_person_count[o] = people_count
    return orientation_to_car_count, orientation_to_person_count


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
                 f'{pan}-{bottom_tilt}-{zoom}' ]
    elif tilt == -30:
        return [ f'{left_horz}-{tilt}-{zoom}', 
                 f'{right_horz}-{tilt}-{zoom}', 
                 current_orientation, 
                 f'{pan}-{top_tilt}-{zoom}']
    return [ f'{left_horz}-{tilt}-{zoom}', 
             f'{right_horz}-{tilt}-{zoom}', 
             current_orientation, 
             f'{pan}-{top_tilt}-{zoom}', 
             f'{pan}-{bottom_tilt}-{zoom}' ]




# set_type is 'train' or 'val'
def generate_dataset(inference_dir, rectlinear_dir, current_frame, orientation_to_frames, set_type, saved_path, project_name):
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
    image_outdir = f'continual-learning/datasets/{project_name}/{set_type}'
    annotations_outdir = f'continual-learning/datasets/{project_name}/annotations'
    os.makedirs(annotations_outdir, exist_ok=True)
    os.makedirs(image_outdir, exist_ok=True)

    print('generating dataset')
    print(orientation_to_frames)
    for o in orientation_to_frames:
        frames = orientation_to_frames[o]
        result_orientation_dir = os.path.join(inference_dir, o)
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

frame_bounds = [(1,1161), (1162,1663), (1664,2823), (2824,3966), (3967, 4983), (4984, 6075), (6076, 7194),  (7195, 7920)]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inference', help='Path to inference dir ')
    ap.add_argument('rectlinear', help='Path to rectlinear dir ')
    
    ap.add_argument('frame_begin', type=int, help='Beginning frame num ')
    ap.add_argument('frame_limit', type=int, help='Ending frame num ')

    ap.add_argument('actual', help='CSV with ground truth couns')

    ap.add_argument('objecttype', help='object type')

    ap.add_argument('weights', help='weights file')
    ap.add_argument('project', help='some unique name for this experiment')

    ap.add_argument('--device', type=int, default=-1)
    args = ap.parse_args()
    object_type = args.objecttype 
    weights_path = args.weights

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()
    gpu = args.device
    use_cuda = gpu >= 0


    data_path = 'continual-learning/datasets/'
    saved_path = 'continual-learning/weights/'

    # Remove old weights 
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)
    os.makedirs(saved_path, exist_ok=True)

    if use_cuda:
        model.cuda(gpu)
        if use_float16:
            model.half()

    frames = []
    frame_to_actual_orientation_to_count = {}
    with open(args.actual, 'r') as f:
        for line in f.readlines():
            orientation_to_count = {}
            items = line.split(',')
            frame_num = 0
            for idx,item in enumerate(items):
                if len(item.strip()) == 0:
                    continue
                if idx == 0: 
                    frame_num = int(item)
                    frames.append(frame_num)
                else:
                    o = item.split(':')[0]
                    count = float(item.split(':')[1])
                    orientation_to_count[o] = count
            frame_to_actual_orientation_to_count[frame_num] = orientation_to_count

    result_idx = -1
    prev_result_idx = -1

    current_fixed_ranks = []
    all_fixed_ranks = []

    current_ranks = []
    all_ranks = []

    current_fixed_accuracies = []
    all_fixed_accuracies = []

    current_accuracies = []
    all_accuracies = []

    # TODO: Add count since last best fixed orientation seen. If count exceeds thresh, visit it
    orientation_to_latest_count = {}

    best_fixed_frames = 0
    total_frames = 0
    best_fixed_orientation = None
    orientation_to_historical_frames = {}
    orientation_to_training_frames = {}
    for current_frame in range(args.frame_begin, args.frame_limit+1):
        if current_frame <= frame_bounds[0][1]:
            result_idx = 0
        elif current_frame <= frame_bounds[1][1]:
            result_idx = 1
        elif current_frame <= frame_bounds[2][1]:
            result_idx = 2
        elif current_frame <= frame_bounds[3][1]:
            result_idx = 3
        elif current_frame <= frame_bounds[4][1]:
            result_idx = 4
        elif current_frame <= frame_bounds[5][1]:
            result_idx = 5
        elif current_frame <= frame_bounds[6][1]:
            result_idx = 6
        elif current_frame <= frame_bounds[7][1]:
            result_idx = 7
        else:
            result_idx = 0 

        sub_frame_begin = frame_bounds[result_idx][0]
        sub_frame_limit = frame_bounds[result_idx][1]

        if result_idx != prev_result_idx:
            print('RESETTING')

            best_fixed_frames = 0
            total_frames = 0

            prev_result_idx = result_idx
            if len(current_accuracies) > 0:
                all_accuracies.append(sum(current_accuracies) / len(current_accuracies))
                current_accuracies.clear()
                all_ranks.append(sum(current_ranks) / len(current_ranks))
                current_ranks.clear() 

                all_fixed_accuracies.append(sum(current_fixed_accuracies) / len(current_fixed_accuracies))
                current_fixed_accuracies.clear()
                all_fixed_ranks.append(sum(current_fixed_ranks) / len(current_fixed_ranks))
                current_fixed_ranks.clear()
               
 
            orientation_to_latest_count.clear()
            orientation_to_count = {}
            # Get best fixed orientation
            for f in range(sub_frame_begin, sub_frame_limit+1):
                if f not in frame_to_actual_orientation_to_count:
                    continue
                elif len(orientation_to_latest_count) == 0:
                    for o in frame_to_actual_orientation_to_count[f]:
                        orientation_to_latest_count[o] = frame_to_actual_orientation_to_count[f][o]
                for o in frame_to_actual_orientation_to_count[f]:
                    if o not in orientation_to_count:
                        orientation_to_count[o] = 0
                    orientation_to_count[o] += frame_to_actual_orientation_to_count[f][o]

            max_count = 0
            for o in orientation_to_count:
                if orientation_to_count[o] > max_count:
                    max_count = orientation_to_count[o]
                    best_fixed_orientation = o

            # Load up historical frames (first 30%)
            orientation_to_historical_frames.clear()
            orientation_to_training_frames.clear()
            for f in range(sub_frame_begin, sub_frame_begin + int(0.3*(sub_frame_limit - sub_frame_begin))):
                if f % SKIP != 0:
                    continue
                for o in orientation_to_count:
                    if o not in orientation_to_historical_frames:
                        orientation_to_historical_frames[o] = []
                    orientation_to_historical_frames[o].append( f)

        if current_frame % SKIP != 0:
            continue

        if current_frame < sub_frame_begin + int(0.3*(sub_frame_limit - sub_frame_begin)):
            if current_frame % SKIP  == 0:
                # Add training set to historical data
                neighboring_orientations = generate_neighboring_orientations(best_fixed_orientation)
                # Iterate through neighboring orientations to get results
                for no in neighboring_orientations:
                    if no not in orientation_to_historical_frames:
                        orientation_to_historical_frames[no] = []
                    orientation_to_historical_frames[no].append(current_frame)
            continue

        print()
        print('Frame ', current_frame)

        print('best fixed orientation ', best_fixed_orientation)

        assert best_fixed_orientation is not None


        actual_orientation_to_count = {}
        neighboring_orientations = generate_neighboring_orientations(best_fixed_orientation)
        # Iterate through neighboring orientations to get results
        for no in neighboring_orientations:
            neighbor_result_orientation_dir = os.path.join(args.inference, no)
            actual_total_cars_list, actual_total_people_list = extract_csv_results(os.path.join(neighbor_result_orientation_dir, f'frame{current_frame}.csv'), CAR_CONFIDENCE_THRESH, PERSON_CONFIDENCE_THRESH)
            if object_type == 'car':
                actual_orientation_to_count[no] = len(actual_total_cars_list)
            elif object_type == 'person':
                actual_orientation_to_count[no] = len(actual_total_people_list)





        # **** AGGREGATION CODE HERE *****
        best_current_orientations = []
        orientation_to_actual_ranking = rank_orientations(actual_orientation_to_count)
        current_model_orientation_to_count = {}

        print('ACTUAL COUNT')
        print(actual_orientation_to_count)
        orientation_to_file = {}
        for o in orientation_to_actual_ranking:
            orientation_to_file[o] = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')
        # Run inference
        orientation_to_car_count, orientation_to_person_count = run_inference(orientation_to_file,  model, args.device)


        print('MODEL COUNT')
        if object_type == 'car':
            orientation_to_model_ranking = rank_orientations(orientation_to_car_count)
            print(orientation_to_car_count)
        elif object_type == 'person':
            orientation_to_model_ranking = rank_orientations(orientation_to_person_count)
            print(orientation_to_person_count)
        orientation_to_latest_ranking = rank_orientations(orientation_to_latest_count)



        # Aggregate ranks
        predicted_orientation_to_ranking = {}
        for o in orientation_to_actual_ranking:
#            predicted_orientation_to_ranking[o] = round( ( orientation_to_model_ranking[o] + orientation_to_latest_ranking[o] ) / 2)
            predicted_orientation_to_ranking[o] = orientation_to_model_ranking[o]

        sorted_dict = {k: v for k, v in sorted(predicted_orientation_to_ranking.items(), key=lambda item: item[1] )}
        for o in sorted_dict:
            if len(best_current_orientations) == 0:
                best_current_orientations.append(o)
            elif sorted_dict[o] == sorted_dict[best_current_orientations[-1]]:
                best_current_orientations.append(o)
            else:
                break



        best_current_orientation = best_current_orientations[0]

        print('Best orientations ', best_current_orientations)

        orientation_to_latest_count[best_current_orientation] = actual_orientation_to_count[best_current_orientation]
        # Add chosen image to potential training data
        if o not in orientation_to_training_frames:
            orientation_to_training_frames[o] = []
        orientation_to_training_frames[o].append( current_frame)



        # Compute stats
        max_count = 0
        for o in actual_orientation_to_count:
            if actual_orientation_to_count[o] > max_count:
                max_count = actual_orientation_to_count[o]
        current_count = actual_orientation_to_count[best_current_orientation]
        fixed_count = actual_orientation_to_count[best_fixed_orientation] 
        fixed_rank = orientation_to_actual_ranking[best_fixed_orientation]
        current_rank = orientation_to_actual_ranking[best_current_orientation]

        print('best fixed orientation ', best_fixed_orientation)
        print('selected orientation ', best_current_orientation)
        if max_count ==  0.0: 
            print('fixed count ', fixed_count, ' / ', max_count, ' = 1.0')
            print('current count ', current_count, ' / ' , max_count, ' = 1.0') 

            current_fixed_ranks.append(1.0)
            current_ranks.append( 1.0)
            current_fixed_accuracies.append(1.0)
            current_accuracies.append( 1.0)
        else:
            print('fixed count ', fixed_count, ' / ', max_count, ' = ', fixed_count / max_count)
            print('current count ', current_count, ' / ' , max_count, ' = ', current_count / max_count)

            current_fixed_accuracies.append(fixed_count / max_count)
            current_accuracies.append( current_count / max_count)

            current_fixed_ranks.append(fixed_rank)
            current_ranks.append( current_rank)


        if max_count == 0.0 or fixed_count / max_count == 1.0:
            best_fixed_frames += 1
        total_frames += 1

        if current_frame % 60 == 0:

           
            # Remove stuff from prior retraining period 
            if os.path.exists(os.path.join(data_path, args.project)):
                shutil.rmtree(os.path.join(data_path, args.project))
            os.makedirs(os.path.join(saved_path, args.project), exist_ok=True)
            os.makedirs(os.path.join(data_path, args.project), exist_ok=True)

            params.train_set  = 'train'
            # Continual learnin

            orientation_to_val_frames = {} 
            len_of_retraining_set = 0 
            len_of_val_set = 0
            # Add newest data to historical data
            for o in orientation_to_training_frames:
                orientation_to_historical_frames[o].extend(orientation_to_training_frames[o])
                len_of_retraining_set += len(orientation_to_training_frames[o])
    
            # Construct retraining set        
            num_retraining_images = 35
            while len_of_retraining_set < num_retraining_images:
                o, _ = random.choice(list(orientation_to_historical_frames.items()))
                new_frame = random.choice(orientation_to_historical_frames[o])
                len_of_retraining_set += 1
                if o not in orientation_to_training_frames:
                    orientation_to_training_frames[o] = []
                orientation_to_training_frames[o].append(new_frame)
    
            num_val_images = 10
            while len_of_val_set < num_val_images:
                o, _ = random.choice(list(orientation_to_historical_frames.items()))
                new_frame = random.choice(orientation_to_historical_frames[o])
    
                len_of_val_set += 1
                if o not in orientation_to_val_frames:
                    orientation_to_val_frames[o] = []
    
                orientation_to_val_frames[o].append(new_frame)
            # Train set
            generate_dataset(args.inference, args.rectlinear, current_frame, orientation_to_training_frames, 'train', saved_path, args.project)
    #        # Val set
            generate_dataset(args.inference, args.rectlinear, current_frame, orientation_to_val_frames, 'val', saved_path, args.project)
            
            weights_path = train.continual_train(params, args.project, weights_path, data_path, saved_path, args.project, num_epochs=20)
            print('Saved weights ', weights_path)
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model.requires_grad_(False)
            orientation_to_training_frames.clear()
    if len(current_accuracies) == 0:
        all_accuracies.append(0.0)
    else:
        all_accuracies.append(sum(current_accuracies) / len(current_accuracies))
    current_accuracies.clear()

    if len(current_fixed_accuracies) == 0:
        all_fixed_accuracies.append(0.0)
    else:
        all_fixed_accuracies.append(sum(current_fixed_accuracies) / len(current_fixed_accuracies))
    current_fixed_accuracies.clear()
    print('Fixed accuracies ', all_fixed_accuracies)
    print(sum(all_fixed_accuracies) / len(all_fixed_accuracies))
    print('Current accuracies ', all_accuracies)
    print(sum(all_accuracies) / len(all_accuracies))

    print()
    print('Fixed ranks ', all_fixed_ranks)
    print(sum(all_fixed_ranks) / len(all_fixed_ranks))

    print('Current ranks ', all_ranks)
    print(sum(all_ranks) / len(all_ranks))


#    print('----')
#    print('durations ', all_best_orientation_durations)
#    print('fixed percentages ', all_best_fixed_percentages)
if __name__ == '__main__':
    main()







