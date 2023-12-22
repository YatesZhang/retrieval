#!/bin/bash

echo "flamingo: {$FLAMINGO}" 
echo "retrieval: {$RETRIEVAL}"
python $RETRIEVAL/tools/gittools/update.py \
    --source $RETRIEVAL \
    --destination $FLAMINGO