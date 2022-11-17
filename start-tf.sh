#! /bin/bash

docker run --name tf-latest --rm -d -t \
-u kstef:kstef \
-v /home/kstef/Semantic-Segmentation-Cityscapes:/home/kstef/Semantic-Segmentation-Cityscapes \
-v /home/kstef/.keras:/home/kstef/.keras \
tensorflow:latest
bash