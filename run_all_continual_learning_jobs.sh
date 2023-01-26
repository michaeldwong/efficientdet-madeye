#!/bin/bash


echo "YOLOv4" > cont_learn_results.txt
python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/yolov4 /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_yolov4_car.csv car logs/madeye-yolov4-both-1-7920/efficientdet-d0_97_20500.pth  cont-learning-project --device 0 >> cont_learn_results.txt

python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/yolov4 /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_yolov4_person.csv person logs/madeye-yolov4-both-1-7920/efficientdet-d0_97_20500.pth  cont-learning-project --device 0 >> cont_learn_results.txt



echo "Tiny YOLOv4" >> cont_learn_results.txt
python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/tiny-yolov4 /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_tiny-yolov4_car.csv car logs/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_94_19962.pth  cont-learning-project --device 0 >> cont_learn_results.txt

python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_tiny-yolov4_person.csv person logs/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_94_19962.pth cont-learning-project --device 0 >> cont_learn_results.txt


echo "SSD" >> cont_learn_results.txt
python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/ssd-voc /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_ssd-voc_car.csv car logs/madeye-ssd-voc-both-1-7920/efficientdet-d0_121_27000.pth  cont-learning-project --device 0 >> cont_learn_results.txt

python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_ssd-voc_person.csv person logs/madeye-ssd-voc-both-1-7920/efficientdet-d0_121_27000.pth  cont-learning-project --device 0 >> cont_learn_results.txt

# Faster RCNN 
echo "Faster RCNN" >> cont_learn_results.txt
python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_faster-rcnn_car.csv car logs/madeye-faster-rcnn-both-1-7920/efficientdet-d0_121_17900.pth  cont-learning-project --device 0 >> cont_learn_results.txt

python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 count-results/7/actual_results_faster-rcnn_person.csv person logs/madeye-faster-rcnn-both-1-7920/efficientdet-d0_121_17900.pth  cont-learning-project --device 0 >> cont_learn_results.txt
