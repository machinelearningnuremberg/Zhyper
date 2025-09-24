#!/bin/bash
MAT_RANK=${1:-8}

for benchmark in boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp
do
    echo "$benchmark"
    bash scripts/start_main.sh mistral "lora_${benchmark}" task "$MAT_RANK"
done