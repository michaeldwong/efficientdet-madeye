


#
#
#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/yolov4_best_fixed_orientations.csv 1 7920 car  input_lists/yolov4-car-inputs.txt
#python3 run_inference.py  -i input_lists/yolov4-car-inputs.txt   -p madeye-yolov4-both-1-7920  -c 0 -w logs/madeye-yolov4-both-1-7920/efficientdet-d0_97_20500.pth --device 1  --output count-results/yolov4-car-count-results.txt
#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/yolov4_best_fixed_orientations.csv 1 7920 person  input_lists/yolov4-person-inputs.txt
#python3 run_inference.py  -i input_lists/yolov4-person-inputs.txt   -p madeye-yolov4-both-1-7920  -c 0 -w logs/madeye-yolov4-both-1-7920/efficientdet-d0_97_20500.pth --device 1  --output count-results/yolov4-person-count-results.txt
#
#
#
#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/ssd_voc_best_fixed_orientations.csv 1 7920 car  input_lists/ssd-voc-car-inputs.txt
#python3 run_inference.py  -i input_lists/ssd-voc-car-inputs.txt   -p madeye-ssd-voc-both-1-7920  -c 0 -w logs/madeye-ssd-voc-both-1-7920/efficientdet-d0_132_29368.pth --device 1  --output count-results/ssd-voc-car-count-results.txt
#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/ssd_voc_best_fixed_orientations.csv 1 7920 person  input_lists/ssd-voc-person-inputs.txt
#python3 run_inference.py  -i input_lists/ssd-voc-person-inputs.txt   -p madeye-ssd-voc-both-1-7920  -c 0 -w logs/madeye-ssd-voc-both-1-7920/efficientdet-d0_132_29368.pth --device 1  --output count-results/ssd-voc-person-count-results.txt
#



#
#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/tiny_yolov4_best_fixed_orientations.csv 1 7920 car  input_lists/tiny-yolov4-car-inputs.txt
#python3 run_inference.py  -i input_lists/tiny-yolov4-car-inputs.txt   -p madeye-tiny-yolov4-both-1-7920  -c 0 -w logs/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_94_19900.pth  --device 1  --output count-results/tiny-yolov4-car-count-results.txt
#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/tiny_yolov4_best_fixed_orientations.csv 1 7920 person  input_lists/tiny-yolov4-person-inputs.txt
#python3 run_inference.py  -i input_lists/tiny-yolov4-person-inputs.txt   -p madeye-tiny-yolov4-both-1-7920  -c 0 -w logs/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_94_19900.pth --device 1  --output count-results/tiny-yolov4-person-count-results.txt
#

#python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/faster_rcnn_best_fixed_orientations.csv 1 7920 person input_lists/faster-rcnn-person-inputs.txt
#python3 run_inference.py  -i input_lists/faster-rcnn-person-inputs.txt   -p madeye-faster-rcnn-both-1-7920  -c 0 -w logs/madeye-faster-rcnn-both-1-7920/efficientdet-d0_140_20600.pth --device 1  --output count-results/faster-rcnn-people-count-results.txt


# -------






python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/yolov4_best_fixed_orientations.csv 1 7920 car  input_lists-4/yolov4-car-inputs.txt
python3 run_inference.py  -i input_lists-4/yolov4-car-inputs.txt   -p madeye-yolov4-both-1-7920  -c 0 -w logs-full-model/madeye-yolov4-both-1-7920/efficientdet-d0_34_59325.pth --device 0  --output count-results-4/yolov4-car-count-results.txt
python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/yolov4_best_fixed_orientations.csv 1 7920 person  input_lists-4/yolov4-person-inputs.txt
python3 run_inference.py  -i input_lists-4/yolov4-person-inputs.txt   -p madeye-yolov4-both-1-7920  -c 0 -w logs-full-model/madeye-yolov4-both-1-7920/efficientdet-d0_34_59325.pth --device 0  --output count-results-4/yolov4-person-count-results.txt



python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/ssd_voc_best_fixed_orientations.csv 1 7920 car  input_lists-4/ssd-voc-car-inputs.txt
python3 run_inference.py  -i input_lists-4/ssd-voc-car-inputs.txt   -p madeye-ssd-voc-both-1-7920  -c 0 -w logs-full-model/madeye-ssd-voc-both-1-7920/efficientdet-d0_33_60418.pth --device 0  --output count-results-4/ssd-voc-car-count-results.txt
python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/ssd_voc_best_fixed_orientations.csv 1 7920 person  input_lists-4/ssd-voc-person-inputs.txt
python3 run_inference.py  -i input_lists-4/ssd-voc-person-inputs.txt   -p madeye-ssd-voc-both-1-7920  -c 0 -w logs-full-model/madeye-ssd-voc-both-1-7920/efficientdet-d0_33_60418.pth --device 0  --output count-results-4/ssd-voc-person-count-results.txt





python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/tiny_yolov4_best_fixed_orientations.csv 1 7920 car  input_lists-4/tiny-yolov4-car-inputs.txt
python3 run_inference.py  -i input_lists-4/tiny-yolov4-car-inputs.txt   -p madeye-tiny-yolov4-both-1-7920  -c 0 -w logs-full-model/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_35_60000.pth  --device 0  --output count-results-4/tiny-yolov4-car-count-results.txt
python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/tiny_yolov4_best_fixed_orientations.csv 1 7920 person  input_lists-4/tiny-yolov4-person-inputs.txt
python3 run_inference.py  -i input_lists-4/tiny-yolov4-person-inputs.txt   -p madeye-tiny-yolov4-both-1-7920  -c 0 -w logs-full-model/madeye-tiny-yolov4-both-1-7920/efficientdet-d0_35_60000.pth --device 0  --output count-results-4/tiny-yolov4-person-count-results.txt


python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/faster_rcnn_best_fixed_orientations.csv 1 7920 car input_lists-4/faster-rcnn-car-inputs.txt
python3 run_inference.py  -i input_lists-4/faster-rcnn-car-inputs.txt   -p madeye-faster-rcnn-both-1-7920  -c 0 -w logs-full-model/madeye-faster-rcnn-both-1-7920/efficientdet-d0_33_40000.pth --device 0  --output count-results-4/faster-rcnn-car-count-results.txt
python3 generate_inference_input_list.py  /scratch/mdwong/rectlinear-output/seattle-dt-1/   ~/Documents/madeye-image-classification/best-fixed-orientation-data/9-27/faster_rcnn_best_fixed_orientations.csv 1 7920 person input_lists-4/faster-rcnn-person-inputs.txt
python3 run_inference.py  -i input_lists-4/faster-rcnn-person-inputs.txt   -p madeye-faster-rcnn-both-1-7920  -c 0 -w logs-full-model/madeye-faster-rcnn-both-1-7920/efficientdet-d0_33_40000.pth --device 0  --output count-results-4/faster-rcnn-people-count-results.txt
