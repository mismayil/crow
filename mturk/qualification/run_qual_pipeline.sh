#!/bin/zsh

python process_cka_qual_results.py --results-path ${1}.csv
python analyze_cka_qual_results.py --results-path ${1}.json
python qualify_cka_qual_workers.py --report-path ${1}_report.json --results-path ${1}.json