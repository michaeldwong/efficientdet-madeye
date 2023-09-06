    

import argparse
import shutil
import pandas as pd
import numpy as np
import cv2
import os
import random

CAR_CONFIDENCE_THRESH = 70.0
PERSON_CONFIDENCE_THRESH = 50.0



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
#    orientations.extend(random.sample(z2_orientations, random.randint(3, 10)))
#    orientations.extend(random.sample(z3_orientations, random.randint(3, 5)))
    return orientations



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

ap = argparse.ArgumentParser()

ap.add_argument('inference', help='Directory of inference results')
ap.add_argument('rectlinear', help='Directory of raw frames')
ap.add_argument('frame_begin', type=int, help='Beginning frame num')
ap.add_argument('frame_limit', type=int, help='Ending frame num')

ap.add_argument('outfile',  type=str, help='Output json file')

args = ap.parse_args()





current_frame = args.frame_begin
frame_bounds = [ (10430, 11988)]
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

        if current_frame % 18 != 0:
            current_frame += 1
            continue
        #######
        # For training on first 20% of frames

        all_orientations =  generate_random_orientations()
        all_orientations = [ '60-0-1', '90-0-1', '90-15-1', '90--15-1', '120-0-1', ]
        orientation_to_car_count = {}
        orientation_to_person_count = {}
        for o in all_orientations:
            infile = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')
            actual_total_cars_list, actual_total_people_list = extract_csv_results(os.path.join(args.inference, o, f'frame{current_frame}.csv'), CAR_CONFIDENCE_THRESH, PERSON_CONFIDENCE_THRESH)
            orientation_to_car_count[o] = len(actual_total_cars_list)
            orientation_to_person_count[o] = len(actual_total_people_list)
        car_ranking = rank_orientations(orientation_to_car_count)
        person_ranking = rank_orientations(orientation_to_person_count)
        for o in all_orientations:
            infile = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')
            f.write(f'{current_frame},{o},{infile},{orientation_to_car_count[o]},{orientation_to_person_count[o]},{car_ranking[o]},{person_ranking[o]}\n')
        current_frame += 1


