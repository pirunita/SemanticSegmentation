NAME="SemanticSegmentation"
PORT=8200
MOUNTED_PATH="/home/cvpr-pu/sungpil2/SemanticSegmentation"

docker run --runtime=nvidia -it --name ${NAME} -v /dev/snd:/dev/snd -v ${MOUNTED_PATH}:/${NAME} -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY \
           -p ${PORT}:${PORT} --ipc=host khosungpil/unv3:2.0

