#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  SAVE_IMAGES=$1; shift
  GPUS=($@)

  # best results on KolektorSDD2 dataset
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH flip_defect_0 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_0_flip_only_defect.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_0 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_0_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_16 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_16_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH flip_defect_16 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_16_flip_only_defect.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}

  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_53 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_53_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH flip_defect_53 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_53_flip_only_defect.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}

  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_126 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_126_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH flip_defect_126 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_126_flip_only_defect.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}

  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_246 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_246_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH flip_defect_246 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_246_flip_only_defect.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}


}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results True "${GPUS[@]}"

