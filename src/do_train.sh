

#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 trainer.py > train.log &
