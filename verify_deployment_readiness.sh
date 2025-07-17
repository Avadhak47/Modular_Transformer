#!/bin/bash
# verify_deployment_readiness.sh - Comprehensive verification for HPC deployment
# Author: Generated for IITD HPC Multi-Node Mathematical Reasoning
# Usage: ./verify_deployment_readiness.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
TOTAL_CHECKS=0

# Check function
check() {
    local description="$1"
    local command="$2"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "Checking $description... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        return 1
    fi
}

# Check with output function
check_with_output() {
    local description="$1"
    local command="$2"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo "Checking $description..."
    
    if output=$(eval "$command" 2>&1); then
        echo -e "${GREEN}✓ PASS${NC}"
        echo "  $output"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  $output"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        return 1
    fi
}

echo -e "${BLUE}"
echo "████████████████████████████████████████████████████████████████"
echo "█                                                              █"
echo "█           HPC Deployment Readiness Verification             █"
echo "█                                                              █"
echo "████████████████████████████████████████████████████████████████"
echo -e "${NC}"

echo -e "\n${YELLOW}=== Core Infrastructure Checks ===${NC}"

# Check main deployment script
check "Main deployment script exists" "test -f deploy.sh"
check "Deploy script is executable" "test -x deploy.sh"

# Check all node configurations
for i in {0..4}; do
    check "Node $i configuration exists" "test -f configs/node_configs/node_${i}_config.json"
    check "Node $i config is valid JSON" "python3 -m json.tool configs/node_configs/node_${i}_config.json"
done

echo -e "\n${YELLOW}=== Training Scripts Checks ===${NC}"

# Check training scripts
check "Multi-node training script exists" "test -f scripts/submit_multi_node_training.sh"
check "Multi-node training script is executable" "test -x scripts/submit_multi_node_training.sh"
check "Node launcher script exists" "test -f scripts/node_training_launcher.sh"
check "Node launcher script is executable" "test -x scripts/node_training_launcher.sh"
check "Monitor dashboard script exists" "test -f scripts/monitor_multi_node_dashboard.sh"
check "Monitor dashboard script is executable" "test -x scripts/monitor_multi_node_dashboard.sh"

echo -e "\n${YELLOW}=== Enhanced Components Checks ===${NC}"

# Check enhanced training components
check "SOTA training module exists" "test -f training/sota_mathematical_reasoning_trainer.py"
check "SOTA data loader exists" "test -f data/sota_math_dataset_loader.py"
check "SOTA evaluation metrics exist" "test -f evaluation/sota_mathematical_metrics.py"

echo -e "\n${YELLOW}=== Container Infrastructure Checks ===${NC}"

# Check container setup
check "Container Dockerfile exists" "test -f containers/math_reasoning.Dockerfile"
check "HPC requirements file exists" "test -f requirements_hpc.txt"

echo -e "\n${YELLOW}=== Documentation Checks ===${NC}"

# Check documentation
check "HPC deployment guide exists" "test -f HPC_MULTI_NODE_DEPLOYMENT_GUIDE.md"
check "README for HPC deployment exists" "test -f README_HPC_DEPLOYMENT.md"
check "Inference guide exists" "test -f INFERENCE_GUIDE.md"

echo -e "\n${YELLOW}=== Python Module Checks ===${NC}"

# Check Python modules can be imported
check "Python 3 available" "which python3"
check "Pip available" "which pip3"

# Check key Python packages (if available)
if command -v python3 >/dev/null 2>&1; then
    check "Torch import test" "python3 -c 'import torch; print(torch.__version__)'"
    check "Transformers import test" "python3 -c 'import transformers; print(transformers.__version__)'"
    check "JSON module test" "python3 -c 'import json'"
    check "OS module test" "python3 -c 'import os'"
fi

echo -e "\n${YELLOW}=== File Structure Validation ===${NC}"

# Check essential directories
check "Source directory exists" "test -d src"
check "Positional encoding modules exist" "test -d src/positional_encoding"
check "Layers directory exists" "test -d src/layers"
check "Utils directory exists" "test -d src/utils"

# Check positional encoding implementations
PE_TYPES=("sinusoidal" "rope" "alibi" "diet" "t5_relative")
for pe in "${PE_TYPES[@]}"; do
    check "Positional encoding $pe exists" "test -f src/positional_encoding/${pe}.py"
done

echo -e "\n${YELLOW}=== Configuration Validation ===${NC}"

# Validate each node configuration has required fields
for i in {0..4}; do
    config_file="configs/node_configs/node_${i}_config.json"
    if [ -f "$config_file" ]; then
        check "Node $i has model_config" "python3 -c 'import json; data=json.load(open(\"$config_file\")); assert \"model_config\" in data'"
        check "Node $i has sota_integration" "python3 -c 'import json; data=json.load(open(\"$config_file\")); assert \"sota_integration\" in data'"
        check "Node $i has training_config" "python3 -c 'import json; data=json.load(open(\"$config_file\")); assert \"training_config\" in data'"
        check "Node $i has hardware_config" "python3 -c 'import json; data=json.load(open(\"$config_file\")); assert \"hardware_config\" in data'"
        
        # Check positional encoding type
        pe_type=$(python3 -c "import json; data=json.load(open('$config_file')); print(data['model_config']['positional_encoding'])")
        check "Node $i PE type ($pe_type) module exists" "test -f src/positional_encoding/${pe_type}.py"
    fi
done

echo -e "\n${YELLOW}=== Summary ===${NC}"

echo "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Failed: ${RED}$CHECKS_FAILED${NC}"

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL CHECKS PASSED! Deployment is ready.${NC}"
    echo -e "\n${BLUE}Next steps:${NC}"
    echo "1. Run: ./deploy.sh full my_experiment_name"
    echo "2. Monitor progress with the dashboard scripts"
    echo "3. Check results in /scratch/\$USER/math_reasoning/experiments/"
    exit 0
else
    echo -e "\n${RED}❌ Some checks failed. Please fix the issues before deployment.${NC}"
    echo -e "\n${YELLOW}Common fixes:${NC}"
    echo "- Install missing Python packages: pip install -r requirements_hpc.txt"
    echo "- Check file permissions: chmod +x *.sh scripts/*.sh"
    echo "- Verify JSON syntax in configuration files"
    exit 1
fi