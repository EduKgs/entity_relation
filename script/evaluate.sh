#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=1 \
--device_id=0 \
--mode=test \
--corpus_type=NYT \
--ensure_corres \
--ensure_rel \
--corres_threshold=0.5 \
--rel_threshold=0.1