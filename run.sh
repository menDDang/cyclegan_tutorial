#!/bin/bash

work_dir="/home/kmyoon/PycharmProjects/CycleGAN"

# step 1. 데이터 준비
if [ ! -d $work_dir/vangogh2photo ]; then
  echo "There is no data set!"
  sh $work_dir/datasets/download_datasets.sh
fi

