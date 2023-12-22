#!/bin/bash

echo "flamingo: {$FLAMINGO}" 
echo "retrieval: {$RETRIEVAL}"
python update.py \
    --source $RETRIEVAL \
    --destination $FLAMINGO