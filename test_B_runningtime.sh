#!/usr/bin/env bash
for algo in 'newmaRFF' 'ScanB'
do
    python test_B_runningtime.py ${algo}
done
