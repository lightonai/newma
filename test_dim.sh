#!/usr/bin/env bash
for algo in 'newmaRFF' 'newmaFF' 'newmaOPU' 'ScanB' 'MA'
do
    python test_dim.py ${algo}
done
