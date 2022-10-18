# !/bin/sh
export CUDA_VISIBLE_DEVICES=1
python3 main.py -c config/spec1.yaml &

export CUDA_VISIBLE_DEVICES=2
python3 main.py -c config/spec2.yaml &

export CUDA_VISIBLE_DEVICES=3
python3 main.py -c config/spec3.yaml 





