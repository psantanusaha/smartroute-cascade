#!/bin/bash
# Quick Start — Run all SmartRoute experiments
# 
# Prerequisites:
#   1. Get a free Groq API key at https://console.groq.com
#   2. export GROQ_API_KEY="your-key-here"
#   3. pip install -r requirements.txt
#
# Estimated total runtime: 30-40 minutes (Groq free tier rate limits)
# 
# TIP: Start with experiment 3 (fastest, ~5 min) to validate classification
#      Then run experiment 1 (~20 min) for the core capability gap data
#      Then run experiment 2 (~10 min) for the business case numbers

set -e

echo "================================================"
echo "SmartRoute Hypothesis Validation"
echo "================================================"
echo ""

if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY not set"
    echo "Get a free key at https://console.groq.com"
    echo "Then: export GROQ_API_KEY='your-key-here'"
    exit 1
fi

mkdir -p results

echo "[1/4] Running Experiment 3: Skill Classification (~5 min)"
echo "     This is fastest — validates if classification works"
python experiment_3_skill_classifier.py
echo ""

echo "[2/4] Running Experiment 1: Capability Gaps (~20 min)"
echo "     Core experiment — maps what each model can/can't do"
python experiment_1_capability_gaps.py
echo ""

echo "[3/4] Running Experiment 2: Traffic Distribution (~10 min)"
echo "     Business case — what % of traffic stays cheap?"
python experiment_2_traffic_distribution.py
echo ""

echo "[4/4] Combined Analysis"
python analyze_results.py
