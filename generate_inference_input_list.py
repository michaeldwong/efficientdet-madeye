
import argparse
import shutil
import pandas as pd
import numpy as np 
import cv2
import os
import random

CAR_CONFIDENCE_THRESH = 70.0
PERSON_CONFIDENCE_THRESH = 50.0




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

ap = argparse.ArgumentParser()

ap.add_argument('inference', help='Directory of inference results')
ap.add_argument('rectlinear', help='Directory of raw frames')
ap.add_argument('fixed_orientations_file', help='CSV with best fixed orientations')
ap.add_argument('frame_begin', type=int, help='Beginning frame num')
ap.add_argument('frame_limit', type=int, help='Ending frame num')

ap.add_argument('objecttype',  type=str, help='Obj type')
ap.add_argument('outfile',  type=str, help='Output json file')

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

        if current_frame % 6 != 0:
            current_frame += 1
            continue
        #######
        # For training on first 20% of frames
        all_orientations = []
        for current_orientation in orientations:
            neighboring_orientations = generate_neighboring_orientations(current_orientation)
            for no in neighboring_orientations:
                if no not in all_orientations:
                    all_orientations.append(no)
        print('current orienttion ', current_orientation)
        print('all orientations ', all_orientations)
        for o in all_orientations:
            infile = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')

            actual_total_cars_list, actual_total_people_list = extract_csv_results(os.path.join(args.inference, o, f'frame{current_frame}.csv'), CAR_CONFIDENCE_THRESH, PERSON_CONFIDENCE_THRESH)
            f.write(f'{current_frame},{o},{infile},{len(actual_total_cars_list)},{len(actual_total_people_list)}\n')
        current_frame += 1

