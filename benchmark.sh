#!/usr/bin/env bash
# benchmark.sh — entry point for /autoresearch loop.
# Runs the active strategy and emits METRIC name=value lines on stdout.
# Reads strategy choice from STRATEGY env var (default: buy_and_hold).
set -e

STRATEGY="${STRATEGY:-buy_and_hold}"
python run_strategy.py --strategy "$STRATEGY"
