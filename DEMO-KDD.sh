#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  SAVE_IMAGES=$1; shift
  GPUS=($@)

    train_KSDD "./datasets/KSDD_FLIP/" $SAVE_IMAGES N_0_FLIP    $RESULTS_PATH 7 33 0  50 1 0.01 1 False  2 1 False False False  "${GPUS[@]}" # Figure 7, Table 4-Row 13
    train_KSDD "./datasets/KSDD/" $SAVE_IMAGES N_0    $RESULTS_PATH 7 33 0  50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7, Table 4-Row 13

    train_KSDD "./datasets/KSDD_FLIP/" $SAVE_IMAGES N_5_FLIP    $RESULTS_PATH 7 33 5  50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7, Table 4-Row 12
    train_KSDD "./datasets/KSDD/" $SAVE_IMAGES N_5    $RESULTS_PATH 7 33 5  50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7, Table 4-Row 12

	train_KSDD "./datasets/KSDD_FLIP/" $SAVE_IMAGES N_10_FLIP   $RESULTS_PATH 7 33 10 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7
	train_KSDD "./datasets/KSDD/" $SAVE_IMAGES N_10   $RESULTS_PATH 7 33 10 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7
    
	train_KSDD "./datasets/KSDD_FLIP/" $SAVE_IMAGES N_15_FLIP   $RESULTS_PATH 7 33 15 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7
	train_KSDD "./datasets/KSDD/" $SAVE_IMAGES N_15   $RESULTS_PATH 7 33 15 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7

    train_KSDD "./datasets/KSDD_FLIP/" $SAVE_IMAGES N_20_FLIP   $RESULTS_PATH 7 33 20 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7
    train_KSDD "./datasets/KSDD/" $SAVE_IMAGES N_20   $RESULTS_PATH 7 33 20 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7

    train_KSDD "./datasets/KSDD_FLIP/" $SAVE_IMAGES N_ALL_FLIP  $RESULTS_PATH 7 33 33 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7
    train_KSDD "./datasets/KSDD/" $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 False  2 1 False False  False  "${GPUS[@]}" # Figure 7

}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results True "${GPUS[@]}"

