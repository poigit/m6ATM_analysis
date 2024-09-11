#!/bin/bash

### please change the following path within <>
FASTQ=<PATH_TO_FASTQ_DIR> # basecalled fastq dir path
BAM=<PATH_TO_SORTED_BAM> # bam file path
REF=<PATH_TO_REF> # reference file path
OUT=<PATH_TO_OUTPUT_DIR> # output dir path

### run m6atm in the "preprocess" mode to get the intermediate files
m6atm run -f $FASTQ -b $BAM -r $REF -o $OUT -Q preprocess

### intermediate files should include "***_data.npy" and "***_label.npy"
### for installation or preparing fastq or bam files, please refer to m6ATM github: https://github.com/poigit/m6ATM