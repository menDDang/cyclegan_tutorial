#!/bin/bash

work_dir="/home/kmyoon/exp/gan/cyclegan"

# step 1. 데이터 준비
if [ ! -d $work_dir/datasets/vangogh2photo ]; then
  echo "There is no data set!"
  bash $work_dir/scripts/download_dataset.sh || exit 1
fi

