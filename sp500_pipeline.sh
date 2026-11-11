#!/bin/bash
#$ -cwd
#$ -N sp500_pipeline
# WRDS Cloud batch job: submit with   qsub sp500_pipeline.sh
# Progress: cat sp500_pipeline.o*    Errors: cat sp500_pipeline.e*
python3 sp500_pipeline.py
