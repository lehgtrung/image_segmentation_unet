#!/usr/bin/env bash

#rsync -aurv src/ zdeploy@10.50.9.11:/data2/zmining/trunglh2/image_segmentation_unet/src

rsync -avz -e "ssh -p 8003" src kyle@72.204.82.59:/home/kyle/trung