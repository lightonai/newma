#!/usr/bin/env bash
for algo in 'newmaRFF' 'newmaFF' 'newmaOPU' 'ScanB' 'MA'
do
    echo ${algo}
    python test_algo_on_data.py ${algo} -n 2000 -nb 500 -d 100 -B 250
done
