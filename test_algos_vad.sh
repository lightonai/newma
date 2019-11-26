#!/usr/bin/env bash
for algo in 'newmaRFF' 'newmaFF' 'newmaOPU' 'ScanB' 'MA'
do
    echo ${algo}
    python test_algo_on_data.py ${algo} -n 1250 -nb 300 -B 150 -dat VAD
done
