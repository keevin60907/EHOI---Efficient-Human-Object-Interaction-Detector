#!/bin/bash

echo "The log file is: " $1
python ensemble.py >> $1
python eval_hybrid.py >> $1