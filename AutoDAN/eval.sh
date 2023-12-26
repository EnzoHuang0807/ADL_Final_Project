#!/bin/sh

python -W ignore eval.py \
  --API_key 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 \
  --low_memory 1 \
  --model vicuna \
  --input_file ./data/AutoDAN.json \
  --AutoDAN \
