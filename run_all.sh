#!/usr/bin/env bash
#run_all.sh

set -e

echo "== Ingest =="
python src/ingest/ingest_data.py

echo

# REMOVED: "== Clean =="
# REMOVED: python src/clean/clean_data.py

echo "== Data Preparation (Clean & Feature Engineering) =="
python src/features/prepare_data.py # NEW SCRIPT NAME
echo

echo "== Train (baseline) =="
python src/model/train_model.py
echo

echo "== Full evaluation =="
python src/eval/full_evaluate.py --model models/rating_model.pkl