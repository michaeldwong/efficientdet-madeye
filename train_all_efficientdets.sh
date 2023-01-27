
#!/bin/bash


python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/yolov4 1 7920  count-datasets/madeye-yolov4-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/yolov4  1 7920  count-datasets/madeye-yolov4-both-1-7920/val.csv validation
python3 train.py -c 0 -p madeye-yolov4-both-1-7920-all-orientations  --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path logs-all-orientations/ --head_only True  

python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/tiny-yolov4 1 7920  count-datasets/madeye-tiny-yolov4-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/tiny-yolov4 1 7920  count-datasets/madeye-tiny-yolov4-both-1-7920/val.csv validation
python3 train.py -c 0 -p madeye-tiny-yolov4-both-1-7920-all-orientations  --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path logs-all-orientations/ --head_only True  

python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/ssd-voc 1 7920  count-datasets/madeye-ssd-voc-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/ssd-voc 1 7920  count-datasets/madeye-ssd-voc-both-1-7920/val.csv validation
python3 train.py -c 0 -p madeye-ssd-voc-both-1-7920-all-orientations  --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path logs-all-orientations/ --head_only True  


python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn  1 7920  count-datasets/madeye-faster-rcnn-both-1-7920/train.csv train
python3 generate_count_dataset.py /scratch/mdwong/rectlinear-output/seattle-dt-1 /scratch/mdwong/inference-frames/seattle-dt-1/faster-rcnn  1 7920  count-datasets/madeye-faster-rcnn-both-1-7920/val.csv validation
python3 train.py -c 0 -p madeye-faster-rcnn-1-7920-all-orientations  --lr 1e-3 --batch_size 4 --load_weights weights/efficientdet-d0.pth  --num_epochs 500 --save_interval 5000  --saved_path logs-all-orientations/ --head_only True  

