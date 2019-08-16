#!/usr/bin/env bash

rsync -avz -e "ssh -p 8003" src/ kyle@72.204.82.59:/home/kyle/trung/image_segmentation_unet/src