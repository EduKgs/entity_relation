#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/train.py \
--ex_index=1 \
--epoch_num=100 \
--device_id=0 \
--corpus_type=NYT \
--ensure_corres \
--ensure_rel

