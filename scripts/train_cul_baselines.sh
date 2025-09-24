#!/bin/bash
MAT_RANK=${1:-8}

for country in Argentina China Egypt France Germany India Italy Japan Mexico Russia Southafrica Turkey UK US
do
    echo "$country"
    bash scripts/start_main.sh mistral "lora_${country}" align "$MAT_RANK"
done