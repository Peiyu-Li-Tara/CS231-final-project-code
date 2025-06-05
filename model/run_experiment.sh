#!/bin/bash

# Usage: 
# ./run_experiment.sh scenario_1.0.py

SCRIPT_PATH="$1"

if [ -z "$SCRIPT_PATH" ]; then
  echo "Usage: $0 path_to_scenario.py"
  exit 1
fi

echo "Running $SCRIPT_PATH..."

# Run the scenario
python "$SCRIPT_PATH"
