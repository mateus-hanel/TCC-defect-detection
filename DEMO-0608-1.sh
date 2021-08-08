#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  SAVE_IMAGES=$1; shift
  GPUS=($@)

  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_53_0608_1 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_53_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH flip_defect_53_0608_1 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_53_flip_only_defect.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}
  
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH baseline_126_0608_1 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_126_new.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}

  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH split_126_synthetic_normal_0608_1 $RESULTS_PATH 15 -1 .\\splits\\KSDD2\\split_126_synthetic_normal.pyb 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}

  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_0_FLIP_DEFECT_0608_1     $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_0_flip_defect.pyb    90 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12

  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_0_0608_1     $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_0_new.pyb    90 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12

  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_50_SYNTHETIC_V2_2    $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_50_synthetic_2.pyb   90 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12


}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results True "${GPUS[@]}"
