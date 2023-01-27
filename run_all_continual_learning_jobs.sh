#!/bin/bash
dir="3"
#
#echo "YOLOv4" > cont_learn_results.txt
#python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/yolov4 /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_yolov4_car.csv car logs/madeye-yolov4-both-1-7920/efficientdet-d0_97_20500.pth  cont-learning-project saved-model-results/continual-learning/${dir}//yolov4_cars.csv --device 1 >> cont_learn_results.txt
#
#python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/yolov4 /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_yolov4_person.csv person logs/madeye-yolov4-both-1-7920/efficientdet-d0_97_20500.pth  cont-learning-project  saved-model-results/continual-learning/${dir}//yolov4_people.csv --device 1 >> cont_learn_results.txt
#
#
#
#echo "Tiny YOLOv4" >> cont_learn_results.txt
#python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/tiny-yolov4 /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_tiny-yolov4_car.csv car logs/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_94_19900.pth  cont-learning-project   saved-model-results/continual-learning/${dir}//tiny_yolov4_cars.csv --device 1 >> cont_learn_results.txt
#
#python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_tiny-yolov4_person.csv person logs/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_94_19900.pth cont-learning-project   saved-model-results/continual-learning/${dir}//tiny_yolov4_people.csv  --device 1 >> cont_learn_results.txt


echo "SSD" >> cont_learn_results.txt
python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/ssd-voc /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_ssd-voc_car.csv car logs/madeye-ssd-voc-both-1-7920/efficientdet-d0_121_27000.pth  cont-learning-project saved-model-results/continual-learning/${dir}//ssd_voc_cars.csv  --device 1 >> cont_learn_results.txt

python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_ssd-voc_person.csv person logs/madeye-ssd-voc-both-1-7920/efficientdet-d0_121_27000.pth  cont-learning-project saved-model-results/continual-learning/${dir}//ssd_voc_people.csv --device 1 >> cont_learn_results.txt

# Faster RCNN 
echo "Faster RCNN" >> cont_learn_results.txt
python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_faster-rcnn_car.csv car logs/madeye-faster-rcnn-both-1-7920/efficientdet-d0_121_17900.pth  cont-learning-project saved-model-results/continual-learning/${dir}//faster_rcnn_cars.csv --device 1 >> cont_learn_results.txt

python3 aggregate_count_results_cont_training.py /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn /scratch/mdwong/rectlinear-output/seattle-dt-1/ 1 7920 ~/Documents/madeye-image-classification/count-results/7/actual_results_faster-rcnn_person.csv person logs/madeye-faster-rcnn-both-1-7920/efficientdet-d0_121_17900.pth  cont-learning-project  saved-model-results/continual-learning/${dir}//faster_rcnn_people.csv   --device 1 >> cont_learn_results.txt
