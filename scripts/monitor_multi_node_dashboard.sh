#!/bin/bash
# scripts/monitor_multi_node_dashboard.sh
# Real-time monitoring dashboard for multi-node mathematical reasoning training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="/scratch/$USER/math_reasoning"
REFRESH_INTERVAL=30  # seconds
LOG_LINES=5

# Function to display header
display_header() {
    clear
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}    IITD HPC Multi-Node Mathematical Reasoning Training${NC}"
    echo -e "${BLUE}                    Real-Time Dashboard${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${GREEN}Last Update: $(date)${NC}"
    echo -e "${GREEN}Base Directory: $BASE_DIR${NC}"
    echo -e "${GREEN}Refresh Interval: ${REFRESH_INTERVAL}s${NC}"
    echo ""
}

# Function to check job status
check_job_status() {
    echo -e "${CYAN}📋 Job Status:${NC}"
    echo "=============="
    
    # Check PBS jobs
    JOB_STATUS=$(qstat -u $USER 2>/dev/null | grep math_reasoning || echo "No active jobs")
    if [[ "$JOB_STATUS" == "No active jobs" ]]; then
        echo -e "${YELLOW}No active mathematical reasoning jobs found${NC}"
    else
        echo "$JOB_STATUS"
    fi
    echo ""
}

# Function to check node status
check_node_status() {
    echo -e "${CYAN}🖥️  Node Status:${NC}"
    echo "==============="
    
    declare -a PE_METHODS=("sinusoidal" "rope" "alibi" "diet" "t5_relative")
    declare -a SOTA_MODELS=("deepseek-math" "internlm-math" "orca-math" "dotamath" "mindstar")
    
    for i in {0..4}; do
        NODE_DIR="$BASE_DIR/results/node_${i}_${PE_METHODS[$i]}_${SOTA_MODELS[$i]}"
        
        if [ -d "$NODE_DIR" ]; then
            # Check if training is running
            LATEST_LOG="$NODE_DIR/logs/training.log"
            if [ -f "$LATEST_LOG" ]; then
                LAST_UPDATE=$(stat -c %Y "$LATEST_LOG" 2>/dev/null || echo 0)
                CURRENT_TIME=$(date +%s)
                TIME_DIFF=$((CURRENT_TIME - LAST_UPDATE))
                
                if [ $TIME_DIFF -lt 300 ]; then  # Less than 5 minutes
                    STATUS="${GREEN}ACTIVE${NC}"
                elif [ $TIME_DIFF -lt 1800 ]; then  # Less than 30 minutes
                    STATUS="${YELLOW}STALE${NC}"
                else
                    STATUS="${RED}INACTIVE${NC}"
                fi
                
                LAST_LINE=$(tail -1 "$LATEST_LOG" 2>/dev/null | cut -c1-60 || echo "No recent activity")
                echo -e "Node $i (${PE_METHODS[$i]}): $STATUS - $LAST_LINE"
            else
                echo -e "Node $i (${PE_METHODS[$i]}): ${RED}NOT_STARTED${NC}"
            fi
        else
            echo -e "Node $i (${PE_METHODS[$i]}): ${RED}NO_OUTPUT_DIR${NC}"
        fi
    done
    echo ""
}

# Function to check GPU utilization
check_gpu_utilization() {
    echo -e "${CYAN}🎮 GPU Utilization:${NC}"
    echo "==================="
    
    if [ -f "$PBS_NODEFILE" ]; then
        for node in $(cat $PBS_NODEFILE | sort | uniq); do
            echo -e "${PURPLE}Node $node:${NC}"
            GPU_INFO=$(ssh $node "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "GPU info unavailable")
            
            if [[ "$GPU_INFO" != "GPU info unavailable" ]]; then
                echo "$GPU_INFO" | while IFS=',' read -r util mem_used mem_total temp; do
                    util=$(echo $util | xargs)
                    mem_used=$(echo $mem_used | xargs)
                    mem_total=$(echo $mem_total | xargs)
                    temp=$(echo $temp | xargs)
                    
                    if [ "$util" -gt 80 ]; then
                        UTIL_COLOR=$GREEN
                    elif [ "$util" -gt 50 ]; then
                        UTIL_COLOR=$YELLOW
                    else
                        UTIL_COLOR=$RED
                    fi
                    
                    echo -e "  GPU: ${UTIL_COLOR}${util}%${NC} | Memory: ${mem_used}/${mem_total}MB | Temp: ${temp}°C"
                done
            else
                echo "  GPU information unavailable"
            fi
        done
    else
        echo "No PBS nodefile found - not running in PBS job"
    fi
    echo ""
}

# Function to check training progress
check_training_progress() {
    echo -e "${CYAN}📈 Training Progress:${NC}"
    echo "===================="
    
    declare -a PE_METHODS=("sinusoidal" "rope" "alibi" "diet" "t5_relative")
    declare -a SOTA_MODELS=("deepseek-math" "internlm-math" "orca-math" "dotamath" "mindstar")
    
    for i in {0..4}; do
        NODE_DIR="$BASE_DIR/results/node_${i}_${PE_METHODS[$i]}_${SOTA_MODELS[$i]}"
        
        # Check for trainer state
        TRAINER_STATE="$NODE_DIR/checkpoints/checkpoint-*/trainer_state.json"
        if ls $TRAINER_STATE 1> /dev/null 2>&1; then
            LATEST_STATE=$(ls -t $TRAINER_STATE | head -1)
            
            if [ -f "$LATEST_STATE" ]; then
                PROGRESS_INFO=$(python3 -c "
import json
try:
    with open('$LATEST_STATE', 'r') as f:
        state = json.load(f)
    
    global_step = state.get('global_step', 0)
    max_steps = state.get('max_steps', 10000)
    progress = (global_step / max_steps * 100) if max_steps > 0 else 0
    
    # Get latest loss from log history
    log_history = state.get('log_history', [])
    latest_loss = 'N/A'
    if log_history:
        for entry in reversed(log_history):
            if 'train_loss' in entry:
                latest_loss = f\"{entry['train_loss']:.4f}\"
                break
    
    print(f'{progress:.1f}%|{global_step}|{max_steps}|{latest_loss}')
except:
    print('0.0%|0|10000|N/A')
" 2>/dev/null)
                
                IFS='|' read -r progress global_step max_steps loss <<< "$PROGRESS_INFO"
                
                if (( $(echo "$progress > 80" | bc -l) )); then
                    PROGRESS_COLOR=$GREEN
                elif (( $(echo "$progress > 50" | bc -l) )); then
                    PROGRESS_COLOR=$YELLOW
                else
                    PROGRESS_COLOR=$RED
                fi
                
                echo -e "Node $i (${PE_METHODS[$i]}): ${PROGRESS_COLOR}${progress}${NC} | Steps: ${global_step}/${max_steps} | Loss: ${loss}"
            else
                echo -e "Node $i (${PE_METHODS[$i]}): ${RED}No state file${NC}"
            fi
        else
            echo -e "Node $i (${PE_METHODS[$i]}): ${RED}Not started${NC}"
        fi
    done
    echo ""
}

# Function to check WandB status
check_wandb_status() {
    echo -e "${CYAN}📊 WandB Tracking:${NC}"
    echo "=================="
    
    # Look for WandB URLs in recent logs
    WANDB_URLS=$(find $BASE_DIR/logs -name "*.out" -o -name "*.log" | xargs grep -h "wandb: " 2>/dev/null | tail -5 || echo "No WandB URLs found")
    
    if [[ "$WANDB_URLS" == "No WandB URLs found" ]]; then
        echo -e "${YELLOW}No active WandB tracking detected${NC}"
    else
        echo "$WANDB_URLS" | while read -r line; do
            if [[ $line == *"wandb:"* ]]; then
                echo "$line"
            fi
        done
    fi
    echo ""
}

# Function to check storage usage
check_storage_usage() {
    echo -e "${CYAN}💾 Storage Usage:${NC}"
    echo "================="
    
    # Check scratch space usage
    QUOTA_INFO=$(lfs quota -hu $USER /scratch 2>/dev/null || echo "Quota information unavailable")
    echo "Scratch space: $QUOTA_INFO"
    
    # Check output directory size
    if [ -d "$BASE_DIR/results" ]; then
        RESULTS_SIZE=$(du -sh "$BASE_DIR/results" 2>/dev/null | cut -f1 || echo "Unknown")
        echo "Results directory size: $RESULTS_SIZE"
    fi
    
    # Check model cache size
    if [ -d "/home/$USER/.cache/huggingface" ]; then
        CACHE_SIZE=$(du -sh "/home/$USER/.cache/huggingface" 2>/dev/null | cut -f1 || echo "Unknown")
        echo "HuggingFace cache size: $CACHE_SIZE"
    fi
    echo ""
}

# Function to show recent errors
check_recent_errors() {
    echo -e "${CYAN}⚠️  Recent Errors:${NC}"
    echo "=================="
    
    # Look for errors in recent logs
    ERROR_COUNT=0
    for log_file in $(find $BASE_DIR/logs -name "*.err" -o -name "*.log" | xargs ls -t | head -10); do
        if [ -f "$log_file" ]; then
            RECENT_ERRORS=$(grep -i "error\|exception\|failed\|cuda.*out of memory" "$log_file" 2>/dev/null | tail -2)
            if [ ! -z "$RECENT_ERRORS" ]; then
                echo -e "${RED}$(basename $log_file):${NC}"
                echo "$RECENT_ERRORS"
                ERROR_COUNT=$((ERROR_COUNT + 1))
            fi
        fi
    done
    
    if [ $ERROR_COUNT -eq 0 ]; then
        echo -e "${GREEN}No recent errors detected${NC}"
    fi
    echo ""
}

# Function to show system resources
check_system_resources() {
    echo -e "${CYAN}⚡ System Resources:${NC}"
    echo "==================="
    
    # CPU and memory usage
    if command -v htop >/dev/null 2>&1; then
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        echo -e "CPU Usage: ${CPU_USAGE}% | Memory Usage: ${MEM_USAGE}%"
    fi
    
    # Network connectivity test
    if ping -c 1 google.com >/dev/null 2>&1; then
        echo -e "Network: ${GREEN}Connected${NC}"
    else
        echo -e "Network: ${RED}Disconnected${NC}"
    fi
    echo ""
}

# Function to show control options
show_controls() {
    echo -e "${PURPLE}📝 Controls:${NC}"
    echo "============"
    echo "Press 'q' to quit, 'r' to refresh now, 'h' for help"
    echo "Ctrl+C to stop monitoring"
    echo ""
}

# Function to show help
show_help() {
    echo -e "${YELLOW}Help - Available Commands:${NC}"
    echo "=========================="
    echo "q - Quit the dashboard"
    echo "r - Refresh immediately"
    echo "h - Show this help"
    echo "l - Show detailed logs"
    echo "s - Show status summary"
    echo "e - Show error details"
    echo ""
    read -p "Press Enter to continue..."
}

# Function to show detailed logs
show_detailed_logs() {
    echo -e "${YELLOW}Detailed Logs:${NC}"
    echo "=============="
    
    declare -a PE_METHODS=("sinusoidal" "rope" "alibi" "diet" "t5_relative")
    
    for i in {0..4}; do
        echo -e "${CYAN}Node $i (${PE_METHODS[$i]}):${NC}"
        LOG_FILE="$BASE_DIR/logs/node_logs/node_${i}_${PE_METHODS[$i]}.log"
        if [ -f "$LOG_FILE" ]; then
            tail -n $LOG_LINES "$LOG_FILE"
        else
            echo "No log file found"
        fi
        echo ""
    done
    
    read -p "Press Enter to continue..."
}

# Main monitoring loop
main_loop() {
    # Set up signal handling
    trap 'echo -e "\n${GREEN}Dashboard stopped.${NC}"; exit 0' SIGINT SIGTERM
    
    while true; do
        display_header
        check_job_status
        check_node_status
        check_gpu_utilization
        check_training_progress
        check_wandb_status
        check_storage_usage
        check_recent_errors
        check_system_resources
        show_controls
        
        # Check for user input with timeout
        if read -t $REFRESH_INTERVAL -n 1 input; then
            case $input in
                q|Q)
                    echo -e "\n${GREEN}Dashboard stopped by user.${NC}"
                    exit 0
                    ;;
                r|R)
                    echo -e "\n${YELLOW}Refreshing...${NC}"
                    continue
                    ;;
                h|H)
                    show_help
                    ;;
                l|L)
                    show_detailed_logs
                    ;;
                s|S)
                    echo -e "\n${YELLOW}Status Summary:${NC}"
                    check_job_status
                    check_node_status
                    read -p "Press Enter to continue..."
                    ;;
                e|E)
                    echo -e "\n${YELLOW}Error Details:${NC}"
                    check_recent_errors
                    read -p "Press Enter to continue..."
                    ;;
            esac
        fi
    done
}

# Check if running in PBS job
if [ -z "$PBS_JOBID" ]; then
    echo -e "${YELLOW}Warning: Not running in a PBS job. Some features may not work.${NC}"
    sleep 2
fi

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}Error: Base directory $BASE_DIR does not exist.${NC}"
    exit 1
fi

# Start the monitoring dashboard
echo -e "${GREEN}Starting Multi-Node Training Dashboard...${NC}"
sleep 1
main_loop