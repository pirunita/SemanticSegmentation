SESSION=1
NAME="DeepLab"
VIS_PORT=8200
ENABLE_VIS=False
MODE="test"
GPU="0"
DATASET="voc"
BASED_MODEL="deeplabv3_resnet101"
CKPT="best_voc_19.pth"

python val.py --session ${SESSION} \
              --name ${NAME} \
              --based_model ${BASED_MODEL} \
              --checkpoint ${CKPT} \
              --vis_port ${VIS_PORT} \
              --gpu_id ${GPU}

