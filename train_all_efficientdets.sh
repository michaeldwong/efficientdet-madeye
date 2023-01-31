
#!/bin/bash


python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/yolov4 2196 16920 datasets/vancouver-yolov4-2196-16920/train  datasets/vancouver-yolov4-2196-16920/annotations/instances_train.csv train
python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/yolov4 2196 16920 datasets/vancouver-yolov4-2196-16920/val  datasets/vancouver-yolov4-2196-16920/annotations/instances_val.csv validation
#python3 train.py -c 0 -p vancouver-yolov4-2196-16920 --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path vancouver-2196-16920/ --head_only True  

python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/tiny-yolov4 2196 16920 datasets/vancouver-tiny-yolov4-2196-16920/train datasets/vancouver-tiny-yolov4-2196-16920/annotations/instances_train.csv train
python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/tiny-yolov4 2196 16920  datasets/vancouver-tiny-yolov4-2196-16920/val datasets/vancouver-tiny-yolov4-2196-16920/annotations/instances_val.csv validation
#python3 train.py -c 0 -p vancouver-tiny-yolov4-2196-16920 --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path vancouver-2196-16920/ --head_only True  

python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/ssd-voc 2196 16920  datasets/vancouver-ssd-voc-2196-16920/train datasets/vancouver-ssd-voc-2196-16920/annotations/instances_train.csv train
python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/ssd-voc 2196 16920  datasets/vancouver-ssd-voc-2196-16920/val datasets/vancouver-ssd-voc-2196-16920/annotations/instances_val.csv validation
#python3 train.py -c 0 -p vancouver-ssd-voc-2196-16920 --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path vancouver-2196-16920/ --head_only True  


python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/faster-rcnn  2196 16920   datasets/vancouver-faster-rcnn-2196-16920/val  datasets/vancouver-faster-rcnn-2196-16920/annotations/instances_train.csv train
python3 generate_dataset.py /disk3/mdwong/rectlinear-output/vancouver /disk3/mdwong/inference-results/vancouver/faster-rcnn  2196 16920   datasets/vancouver-faster-rcnn-2196-16920/val  datasets/vancouver-faster-rcnn-2196-16920/annotations/instances_val.csv validation
#python3 train.py -c 0 -p vancouver-faster-rcnn-2196-16920 --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path vancouver-2196-16920/ --head_only True  

