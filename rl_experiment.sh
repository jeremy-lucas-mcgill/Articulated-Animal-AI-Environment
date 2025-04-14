#!/bin/bash
# Usage: ./run_tmux.sh
#
# This script launches multiple training sessions in separate tmux windows.
# Each run is specified as "MODEL,RunName,PRESET".
# Preset parameters are defined in an associative array.

####################################
# Configurable Variables
####################################
MAX_STEPS=3000 * 5 # 5 frames per action
N_STEPS=1000 * $MAX_STEPS # N_STEPS = number of episodes * MAX_STEPS
USE_RAYCASTS="--use-raycasts"    # Leave empty ("") if you don't want to use raycasts.
N_CHECKPOINTS=200 # to use specific way of saving checkpoints modify script trainSB3Model.py
N_EVALS=200 # to use specific way of saving checkpoints modify script trainSB3Model.py
NO_GRAPHICS="--no-graphics"
# Path to the Unity environment executable
PATH_TO_ENV="" # full path to the Unity environment executable

####################################
# List of runs (MODEL,RunName,PRESET)
####################################
# Format: "MODEL,RunName,PRESET"
RUNS=( 
    "PPO,Experiment,L0R9" 
    "PPO,Experiment,L0R9" 
    "PPO,Experiment,L0R9" 
    "PPO,Experiment,L0R9" 
    # "PPO,Experiment1,L0" 
    # "PPO,Experiment2,L0" 
    # "PPO,Experiment3,L0" 
    # "PPO,Experiment1,LR5" 
    # "PPO,Experiment1,LR9" 
    # "A2C,Experiment2,L0" 
    # "TD3,Experiment3,LR9" 
    )

####################################
# Associative array for preset parameters
####################################
# Ensure your shell supports associative arrays (bash 4+).
declare -A PRESETS_MAP
PRESETS_MAP["LR5"]="--LR-freq 1 --LR-minDiff 5 --LR-maxDiff 5"
PRESETS_MAP["LR9"]="--LR-freq 1 --LR-minDiff 9 --LR-maxDiff 9"
PRESETS_MAP["LR1"]="--LR-freq 1 --LR-minDiff 1 --LR-maxDiff 1"
PRESETS_MAP["L0R9"]="--LR-freq 0.5 --LR-maxDiff 9 --L0-freq 0.5 --L0-maxDiff 0"
PRESETS_MAP["L0"]="--L0-freq 1"
# Add more presets here as needed:
# PRESETS_MAP["YourPresetName"]="--flag1 value1 --flag2 value2 ..."

####################################
# Training command template
####################################
# Placeholders:
#   first %s  -> model
#   second %s -> preset parameters
#   third %s  -> combined run name (run name with preset tag)
TRAIN_COMMAND="python3 trainSB3Model.py --model %s %s --max-step ${MAX_STEPS} --n-steps ${N_STEPS} ${USE_RAYCASTS} --name %s --n-checkpoints ${N_CHECKPOINTS} --n-evals ${N_EVALS} ${NO_GRAPHICS} --path \"${PATH_TO_ENV}\""

####################################
# Environment setup
####################################

# Name of the tmux session
SESSION_NAME="rl_training"

# Create a new detached tmux session (first window will be created automatically)
tmux new-session -d -s "$SESSION_NAME" -n "window0"

# Window counter for tmux window naming
window_index=0

####################################
# Loop through the RUNS array and create a window for each run
####################################
for tuple in "${RUNS[@]}"; do
    # Split the tuple into model, run_name, and preset using comma as delimiter
    IFS=',' read -r model run_name preset_name <<< "$tuple"
    
    # Retrieve the preset parameters from the associative array
    preset_params="${PRESETS_MAP[$preset_name]}"
    if [ -z "$preset_params" ]; then
        echo "Warning: Preset '$preset_name' not found. Using no extra preset parameters."
        preset_params=""
    fi

    # Create a combined run name that indicates the preset used
    combined_run_name="${run_name}_${preset_name}"
    
    # Format the training command with the current model, preset parameters, and combined run name
    formatted_command=$(printf "$TRAIN_COMMAND" "$model" "$preset_params" "$combined_run_name")
    
    # If it's not the first window, create a new window.
    if [ $window_index -ne 0 ]; then
        tmux new-window -t "$SESSION_NAME" -n "window${window_index}"
    fi


    # init the venv
    tmux send-keys -t "$SESSION_NAME:$window_index" "conda init" C-m
    tmux send-keys -t "$SESSION_NAME:$window_index" "source ~/.bashrc" C-m
    tmux send-keys -t "$SESSION_NAME:$window_index" "conda activate ./rl-research-env" C-m

    # change the dir for calling script
    # tmux send-keys -t "$SESSION_NAME:$window_index" "cd Articulated-Animal-AI-Environment/" C-m
    
    
    # run the script
    tmux send-keys -t "$SESSION_NAME:$window_index" "$formatted_command" C-m


    # Increment the window index for the next command.
    window_index=$((window_index+1))
done

# Attach to the tmux session so you can monitor all runs.
tmux attach-session -t "$SESSION_NAME"
