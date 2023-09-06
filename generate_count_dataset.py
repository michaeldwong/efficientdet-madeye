import argparse
import os 
import pandas as pd
import random

CAR_CONFIDENCE_THRESH = 70.0
PERSON_CONFIDENCE_THRESH = 50.0





def generate_all_orientations():
    orientations = []
    # r1 controls horizontal rotation. r1 = 0 means center point of 0.5
    # r2 contorls vertical rotation. r2 = 0 is focused on the ground, r2 = 90 is straight
    for r1 in range(0,360,30):
        for r2 in  [ -30, -15, 0, 15, 30]:
            for z in [1,2,3]:
                orientations.append(f'{r1}-{r2}-{z}')
    return orientations

def generate_random_orientations(inference_file):
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

    # Hacky way to add zoom for poor performing models
    if 'ssd-voc' in inference_file :
        orientations.extend(random.sample(z1_orientations, random.randint(8, 20)))
        orientations.extend(random.sample(z2_orientations, random.randint(10, 30)))
        orientations.extend(random.sample(z3_orientations, random.randint(8, 12)))
    if 'tiny-yolov4' in inference_file:
        orientations.extend(random.sample(z1_orientations, random.randint(10, 22)))
        orientations.extend(random.sample(z2_orientations, random.randint(10, 22)))
        orientations.extend(random.sample(z3_orientations, random.randint(5, 8)))
    else:
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


def main():
    
    ap = argparse.ArgumentParser()

    ap.add_argument('inference_dir', type=str, help='Inference dir (include model)')
    ap.add_argument('rectlinear', type=str, help='Rectlinear dir')

    ap.add_argument('object', type=str, help='\'car\' or \'person\'')
    ap.add_argument('frame_begin', type=int, help='Output weights file')
    ap.add_argument('frame_limit', type=int, help='Output weights file')
    ap.add_argument('outfile', type=str, help='Output annotations file')

    ap.add_argument('dataset_type', type=str, help='\'train\' or \'valdiation\'')
    args = ap.parse_args()

    current_frame = args.frame_begin
    with open(args.outfile, 'w') as f_annotations:
        while current_frame <= args.frame_limit:
            if current_frame % 6 != 0:
                current_frame += 1
                continue
            
            # For training on first 20% of frames
            total_frames = int((args.frame_limit - args.frame_begin) * 0.5)
            val_upper_bound = int((args.frame_limit - args.frame_begin) * 0.7)
            if args.dataset_type == 'validation':
                if current_frame <=  args.frame_begin + total_frames or current_frame >=  args.frame_begin + val_upper_bound:
                    current_frame += 1
                    continue

            elif args.dataset_type == 'train':
                if current_frame >= args.frame_begin + total_frames:
                    current_frame += 1
                    continue

            for o in generate_random_orientations(args.inference_dir):
                neighbor_result_orientation_dir = os.path.join(args.inference_dir, o)
                inference_file = os.path.join(neighbor_result_orientation_dir, f'frame{current_frame}.csv')
                if not os.path.exists(inference_file):
                    print(inference_file, ' DNE')
                    continue
                if os.path.getsize(inference_file) > 0:
                    orientation_df = pd.read_csv(inference_file) 
                    orientation_df.columns = ['left', 'top', 'right', 'bottom', 'class', 'confidence']

                    car_count = 0
                    people_count = 0

                    for idx, row in orientation_df.iterrows():
                        if row['class'] == 'car' and row['confidence'] >= CAR_CONFIDENCE_THRESH:
                            car_count += 1
                        if row['class'] == 'person' and row['confidence'] >= PERSON_CONFIDENCE_THRESH:
                            people_count += 1

                    orig_image_file = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')
                    if args.object.strip() == 'car':
                        f_annotations.write(orig_image_file + ',' + str(car_count) + '\n')
                    elif args.object.strip() == 'person':
                        f_annotations.write(orig_image_file + ',' + str(people_count) + '\n')
                    else:
                        print('Invalid object type ', args.object)
                        exit()

            current_frame += 1




if __name__ == '__main__':
    main()
