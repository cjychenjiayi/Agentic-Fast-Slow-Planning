#!/bin/bash

# Path to your CARLA main directory
BASE_DIR="Your_carla_file"        
AUTO_DIR="$BASE_DIR/auto_fetch_bbox"

# Path to your output directory
OUTPUT_BASE_DIR="Your_output_dir" 
LOG_FILE="$AUTO_DIR/process_log.txt"

start_carla() {
    echo "Starting CARLA..."
    export RANDFILE="/dev/urandom"
    sh "$BASE_DIR/CarlaUE4.sh" -quality-level=Epic &
    CARLA_PID=$!
    if ! ps -p $CARLA_PID > /dev/null; then
        echo "Failed to start CARLA process."
        exit 1
    fi
    sleep 20
}

stop_carla() {
    echo "Stopping CARLA..."
    if kill -0 $CARLA_PID 2>/dev/null; then
        kill -9 $CARLA_PID
        wait $CARLA_PID 2>/dev/null
        while kill -0 $CARLA_PID 2>/dev/null; do
            sleep 2
        done
        echo "CARLA stopped."
    else
        echo "CARLA process is not running."
    fi
}

run_select_bp() {
    cd "$AUTO_DIR"
    python select_bps.py
}

run_collect_datas() {
    cd "$AUTO_DIR"
    python collect_datas.py --output_dir "$1" --map_name "$2" &
    COLLECT_PID=$!
    
    wait $COLLECT_PID
    echo "collect_datas.py stopped."
}

mkdir -p "$OUTPUT_BASE_DIR"

map_names=("Town10HD" "Town05" "Town03" "Town02" "Town04" "Town06" "Town07")

SEED=$(date +%s)
echo "SEED: $SEED"

RANDOM=$SEED
echo $RANDOM
echo $RANDOM
echo $RANDOM

for map_name in "${map_names[@]}"; do
    mkdir -p "${OUTPUT_BASE_DIR}/${map_name}"
done

count=0
while true; do
    if [[ -f "$AUTO_DIR/control.txt" ]] && [[ $(tr -d '[:space:]' < "$AUTO_DIR/control.txt") == "stop" ]]; then
        echo "Stop command detected. Exiting loop..."
        break
    fi
    
    run_select_bp
    
    map_length=${#map_names[@]}
    map_index=$(( count % map_length ))
    map_name="${map_names[$map_index]}"
    output_dir="${OUTPUT_BASE_DIR}/${map_name}"

    for ((j=0; j<10; j++)); do
        start_carla
        run_collect_datas "$output_dir" "$map_name"
        stop_carla
    done

    ((count+=1))

    if (( count % 10 == 0 )); then
        current_time=$(date +"%Y-%m-%d %H:%M:%S")
        echo "$current_time - Processed $((count * 10)) iterations." >> "$LOG_FILE"
    fi
    
    sleep 4
done

echo "Script finished."
