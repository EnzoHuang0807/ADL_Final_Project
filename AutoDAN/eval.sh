#!/bin/sh

python -W ignore eval.py \
  --API_key \
  --low_memory 1 \
  --model vicuna \
  --input_file ./data/AutoDAN.json \
  --AutoDAN \
