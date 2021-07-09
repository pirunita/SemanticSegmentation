---
title : Semantic Segmentation
version : 1.0
writer: khosungpil
type : Version document
objective : Practice
---

# Environment #
* OS: ubuntu 16.04
* CPU Resource: Inter(R) Core(TM) i7-6700 CPU @ 3.40GHz
* GPU Resource: GTX 1080ti 2x
* Docker Version: 19.03.8

# Usage #
## Requirement ## 
* Recommend using nvidia-docker

### Docker version ###
1. xhost local:root
2. Edit MOUNTED_PATH where code file is in docker_setting.sh 
3. Run docker_setting.sh
~~~
bash docker_setting.sh
~~~

### Pretrained model ###
model | link
------|----------
FCN8s | <a href="https://www.dropbox.com/s/rmkwkc5ra2v8orh/FCN.pth?dl=0">[Download]</a>
DeepLabV3 | <a href="https://www.dropbox.com/s/rjqinur76pa89qu/DeepLab.pth?dl=0">[Download]</a>