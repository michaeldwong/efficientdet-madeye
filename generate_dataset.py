import json
import argparse
import shutil
import pandas as pd
import numpy as np 
import cv2
import os
import random

CAR_CONFIDENCE_THRESH = 70.0
PERSON_CONFIDENCE_THRESH = 50.0
SKIP = 6

def generate_all_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    for r1 in range(0,360,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            for z in [1,2,3]:
                orientations.append(f'{r1}-{r2}-{z}')
    return orientations

def generate_random_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    z1_orientations = []
    z2_orientations = []
    z3_orientations = []
    for r1 in range(0,360,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            z1_orientations.append(f'{r1}-{r2}-1')
            z2_orientations.append(f'{r1}-{r2}-2')
            z3_orientations.append(f'{r1}-{r2}-3')
    orientations.extend(random.sample(z1_orientations, random.randint(10, 30)))
    orientations.extend(random.sample(z2_orientations, random.randint(3, 10)))
    orientations.extend(random.sample(z3_orientations, random.randint(3, 5)))
    return orientations


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
        

ap = argparse.ArgumentParser()
ap.add_argument('rectlinear', help='Directory of raw frames')
ap.add_argument('inference_dir', help='Directory of inference results (include model folder)')
ap.add_argument('fixed_orientations_file', help='CSV with best fixed orientations')
ap.add_argument('frame_begin', type=int, help='Beginning frame num')
ap.add_argument('frame_limit', type=int, help='Ending frame num')
ap.add_argument('objecttype', type=str, help='car or person')
ap.add_argument('outdir',  type=str, help='Output directory for images')
ap.add_argument('outfile',  type=str, help='Output json file')
ap.add_argument('dataset_type',  type=str, help='\'train\' or \'validation\'')

ap.add_argument('--ignore-begin', default=0, type=int, help='Beginnign frame num to ignore')  
ap.add_argument('--ignore-limit', default=0, type=int, help='Ending frame num to ignore')  
ap.add_argument("--per-orientation", action="store_true") 
args = ap.parse_args()


object_type = args.objecttype

best_fixed_df = pd.read_csv(args.fixed_orientations_file)

frame_limit_to_orientation = {}
for idx, row in best_fixed_df.iterrows():
    if (object_type == 'car' or object_type == 'both') and row['class'] == 'car':
        if row['frame_limit'] not in frame_limit_to_orientation:
            frame_limit_to_orientation[row['frame_limit']] = []  
        if row['orientation'] not in frame_limit_to_orientation[row['frame_limit']]:
            frame_limit_to_orientation[row['frame_limit']].append(row['orientation'])
    elif (object_type == 'person' or object_type == 'both') and row['class'] == 'person':
        if row['frame_limit'] not in frame_limit_to_orientation:
            frame_limit_to_orientation[row['frame_limit']] = []
        if row['orientation'] not in frame_limit_to_orientation[row['frame_limit']]:
            frame_limit_to_orientation[row['frame_limit']].append( row['orientation'])

current_frame = args.frame_begin
frame_bounds = [(1,1161), (1162,1663), (1664,2823), (2824,3966), (3967, 4983), (4984, 6075), (6076, 7194),  (7195, 7920), (16939,18418)]
current_frame = args.frame_begin
frames_added = []
orientation_to_avg_count_list = {}

result_idx = -1
prev_result_idx = -1




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

annotation_id = 0
image_id = 0
print('Generating daataset for ', args.outfile)
with open(args.outfile, 'w') as f:
    while current_frame <= args.frame_limit:
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
        elif current_frame <= frame_bounds[8][1]:
            result_idx = 8
        else:
            result_idx = 0 
        sub_frame_begin = frame_bounds[result_idx][0]
        sub_frame_limit = frame_bounds[result_idx][1]
        orientations = frame_limit_to_orientation[sub_frame_limit][:1]

        if current_frame % SKIP != 0:
            current_frame += 1
            continue
        if current_frame >= args.ignore_begin and current_frame <= args.ignore_limit:
            current_frame += 1
            continue
        #######
        # For training on first 20% of frames
        total_frames = int((frame_bounds[result_idx][1] - frame_bounds[result_idx][0]) * 0.3)

        val_upper_bound = int((frame_bounds[result_idx][1] - frame_bounds[result_idx][0]) * 0.4)
        if args.dataset_type == 'validation':
            if current_frame <=  frame_bounds[result_idx][0] + total_frames or current_frame >=  frame_bounds[result_idx][0] + val_upper_bound:
                current_frame += 1
                continue

        elif args.dataset_type == 'train':
            if current_frame >= frame_bounds[result_idx][0] + total_frames:
                current_frame += 1
                continue

#        if current_frame >= frame_bounds[result_idx][0] + total_frames:
#            current_frame += 1
#            continue
        



        ######
#        if current_frame % 2 != 0:
#            current_frame += 1
#            continue
        ######
        # FOr training with 66% (dispersed) of training set
#        frames_added.append(current_frame)
#        if len(frames_added) >= 3:
#            current_frame += 1
#            frames_added.clear()
#            continue
#        ######


#        if int(orientations[0][:orientations[0].index('-')]) % 60 != 0:
#            current_frame += 1
#            continue
#        all_orientations = []
#        for current_orientation in orientations:
#            neighboring_orientations = generate_neighboring_orientations(current_orientation)
#            for no in neighboring_orientations:
#                if no not in all_orientations:
#                    all_orientations.append(no)
#        for o in all_orientations:
        for o in generate_random_orientations():
            neighbor_result_orientation_dir = os.path.join(args.inference_dir, o)
            inference_file = os.path.join(neighbor_result_orientation_dir, f'frame{current_frame}.csv')
            if os.path.getsize(inference_file) > 0:
                orientation_df = pd.read_csv(inference_file)
                orientation_df.columns = ['left', 'top', 'right', 'bottom', 'class', 'confidence']
                orig_image_file = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')
                image_file = f'{o}-frame{current_frame}.jpg'
                dest = f'{args.outdir}/{image_file}'
                shutil.copy(orig_image_file, dest)
                json_dict['images'].append({"id": image_id, "file_name": image_file, "width": 1280, "height": 720, "date_captured": "", "license": 1, "coco_url": "", "flickr_url": ""})
                annotation_id = create_annotations(f, image_file, orientation_df, o, object_type, json_dict, image_id, annotation_id)
            image_id += 1

        current_frame += 1

    json.dump(json_dict, f)
