#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  SAVE_IMAGES=$1; shift
  GPUS=($@)

  # best results on KolektorSDD2 dataset
  #train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_0_FLIP     $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_0_flip.pyb    90 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12
  #train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_10_FLIP    $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_10_flip.pyb   90 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_50_FLIP    $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_50_flip.pyb   60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_150_FLIP   $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_150_flip.pyb  60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_300_FLIP   $RESULTS_PATH 1 300  .\\splits\\STEEL\\split_300_300_flip.pyb  60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12

}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results True "${GPUS[@]}"
