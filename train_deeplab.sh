SESSION=1
NAME="DeepLab"
VIS_PORT=8200
MODE="train"
GPU="0,1"
TRAIN_BATCH_SIZE=8
DATASET="voc"
BASED_MODEL="deeplabv3_resnet101"

python train_deeplab.py --session ${SESSION} \
                --name ${NAME} \
                --mode ${MODE} \
                --based_model ${BASED_MODEL} \
                --vis_port ${VIS_PORT} \
                --batch_size ${TRAIN_BATCH_SIZE} \
                --gpu_id ${GPU}