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

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--inputs', type=str, help='Input txt file with full names of files to run inference on')
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('-o', '--output', type=str, default='count-results.txt', help='output file')

ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
args = ap.parse_args()

input_file = args.inputs
compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(image_paths,  model, threshold=0.05):

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    with open(args.output, 'w') as f_results:
        # In format frame,orientation,file
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                items = line.strip().split(',')
                frame = int(items[0])
                orientation = items[1]
                image_path = items[2]

                print('Processing ', image_path)
                ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
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
                    f_results.write(f'{frame},{orientation},{car_count},{people_count}\n')
                    continue
                for i in range(0,len(scores)):
                    # Class id 0 is car, 1 is person
                    if scores[i] >= 0.7  and class_ids[i] == 0 :
                        car_count += 1
                    elif scores[i] >= 0.5 and class_ids[i] == 1:
                        people_count += 1

                print('car count ', car_count, ' person count ', people_count)
                f_results.write(f'{frame},{orientation},{car_count},{people_count}\n')


if __name__ == '__main__':

    image_paths = []
    
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    torch.save(model.backbone_net.state_dict(), os.path.join('weights', 'efficientnet-d0-counter.pth'))
    exit()
    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()

    evaluate_coco(input_file, model)

