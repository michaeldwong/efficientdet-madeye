#!/bin/bash


python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/yolov4 1 7920  count-datasets/madeye-yolov4-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/yolov4  1 7920  count-datasets/madeye-yolov4-both-1-7920/val.csv validation
#python3 test_efficientnet.py -i single_vid_train.csv -v single_vid_val.csv  -b weights/efficientnet-d0-backbone.pth -s count-weights/madeye-yolov4-both-1-7920

python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/tiny-yolov4 1 7920  count-datasets/madeye-tiny-yolov4-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/tiny-yolov4 1 7920  count-datasets/madeye-tiny-yolov4-both-1-7920/val.csv validation
#python3 test_efficientnet.py -i single_vid_train.csv -v single_vid_val.csv  -b weights/efficientnet-d0-backbone.pth -s count-weights/madeye-tiny-yolov4-both-1-7920

python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/ssd-voc 1 7920  count-datasets/madeye-ssd-voc-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/ssd-voc 1 7920  count-datasets/madeye-ssd-voc-both-1-7920/val.csv validation
#python3 test_efficientnet.py -i single_vid_train.csv -v single_vid_val.csv  -b weights/efficientnet-d0-backbone.pth -s count-weights/madeye-ssd-voc-both-1-7920


python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn  1 7920  count-datasets/madeye-faster-rcnn-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn  1 7920  count-datasets/madeye-faster-rcnn-both-1-7920/val.csv validation
#python3 test_efficientnet.py -i single_vid_train.csv -v single_vid_val.csv  -b weights/efficientnet-d0-backbone.pth -s count-weights/madeye-faster-rcnn-both-1-7920


