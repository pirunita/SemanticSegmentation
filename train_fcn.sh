SESSION=0
NAME="FCN"
VIS_PORT=8200
SIZE=336
MODE="train"
GPU="0"
TRAIN_BATCH_SIZE=8
DATASET="voc"
BASED_MODEL="fcn8s_vgg16"

python train_fcn.py --session ${SESSION} \
                --name ${NAME} \
                --mode ${MODE} \
                --based_model ${BASED_MODEL} \
                --vis_port ${VIS_PORT} \
                --batch_size ${TRAIN_BATCH_SIZE} \
                --gpu_id ${GPU} \
                --crop_size ${SIZE}