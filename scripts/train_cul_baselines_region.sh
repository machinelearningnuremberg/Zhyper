#!/bin/bash
MAT_RANK=${1:-8}

for region in Europe Asia Africa MiddleEast Latinamerica
do
    echo "$region"
    bash scripts/start_main.sh mistral "lora_${region}" align "$MAT_RANK"
done