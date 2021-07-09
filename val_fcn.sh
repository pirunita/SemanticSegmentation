SESSION=0
NAME="FCN"
VIS_PORT=8200
MODE="test"
GPU="0"
DATASET="voc"
BASED_MODEL="fcn8s_vgg16"
CKPT="best_voc_17.pth"

python val.py --session ${SESSION} \
              --name ${NAME} \
              --based_model ${BASED_MODEL} \
              --checkpoint ${CKPT} \
              --vis_port ${VIS_PORT} \
              --gpu_id ${GPU}