# lumbar
Spark“数字人体”AI挑战赛 ——脊柱疾病智能诊断大赛


cd code/
python anno.py

nohup ./tools/dist_train.sh ./mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py 2 > log 2>&1 &
python ./mmdetection/tools/test.py ./mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py ./mmdetection/work_dirs/x/epoch_80.pth --out ../data/results_bbox_test.pkl

nohup ./tools/dist_train.sh ./mmpose/configs/top_down/hrnet/coco/hrnet_w32_coco_384x288.py 2 > log 2>&1 &

python pose.py
python submit.py