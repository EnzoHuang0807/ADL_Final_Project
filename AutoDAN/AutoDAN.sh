#!/bin/sh

python -W ignore AutoDAN.py \
  --API_key  \
  --low_memory 1 \
  --model vicuna \
  --input_file ./data/similar_pair.json \
  --output_file ./data/AutoDAN_1.json \
  --log_file ./data/log_1.json
