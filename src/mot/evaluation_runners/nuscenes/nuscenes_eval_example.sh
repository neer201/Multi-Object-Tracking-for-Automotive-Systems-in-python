python3 src/mot/evaluation_runners/nuscenes/nuscenes_process.py --config-file='./src/mot/evaluation_runners/nuscenes/config.yml' --detection-file='./src/mot/evaluation_runners/nuscenes/data/detections/megvii_val.json'
python3 src/mot/evaluation_runners/nuscenes/evaluate.py ./results.json --version='v1.0-mini' --dataroot='./data/nuscenes/dataset' --eval_set='mini_val'
