#!/bin/bash

### m6Anet
OUT=/PATH/TO/OUTPUT/DIR
FAST5=/PATH/TO/FAST5
FASTQDIR=/PATH/TO/FASTQ
REF=/PATH/TO/REF
JOB=m6anet

# basecalling
guppy_basecaller -c rna_r9.4.1_70bps_hac.cfg -i $FAST5 -s $FASTQDIR -r --compress_fastq
cat $FASTQDIR/*.fastq.gz > $FASTQDIR/pass.fastq.gz
FASTQ=$FASTQDIR/pass.fastq.gz
FASTQSUM=$FASTQDIR/sequencing_summary.txt

# mapping
minimap2 -ax map-ont -uf --secondary=no $REF $FASTQ > $OUT/$JOB.sam
samtools view -Sb $OUT/$JOB.sam > $OUT/$JOB.bam
samtools sort $OUT/$JOB.bam -o $OUT/${JOB}_sorted.bam
samtools index $OUT/${JOB}_sorted.bam
BAM=$OUT/${JOB}_sorted.bam

# nanopolish
nanopolish index -s $FASTQSUM -d $FAST5 $FASTQ
nanopolish eventalign \
--reads $FASTQ --bam $BAM --genome $REF \
--signal-index --scale-events --summary $OUT/${JOB}_summary.txt > $OUT/${JOB}_events.tsv

# m6anet
m6anet dataprep --eventalign $OUT/${JOB}_events.tsv --out_dir $OUT
m6anet inference --input_dir $OUT --out_dir $OUT --num_iterations 1000


### m6ABasecaller
FAST5=/PATH/TO/FAST5
FASTQ=/PATH/TO/FASTQ
OUT=/PATH/TO/OUTPUT
MODEL=/PATH/TO/MODEL/rna_r9.4.1_70bps_m6A_hac.cfg
REF=/PATH/TO/REF

guppy_basecaller -i $FAST5 -s $FASTQ -c $MODEL --fast5_out -r
/PATH/TO/modPhred/run -f $REF -o $OUT -i $FASTQ/workspace --minModFreq 0


### Tombo
FAST5=/PATH/TO/MULTI_FAST5
OUT=/PATH/TO/OUTPUT
FAST5_S=$OUT/fast5
FASTQ=$OUT/fastq
REF=/PATH/TO/REF
JOB=tombo

# single fast5
multi_to_single_fast5 -i $FAST5 -s $FAST5_S --recursive
guppy_basecaller -i $FAST5_S -s $FASTQ -c rna_r9.4.1_70bps_hac.cfg --fast5_out -r
cat $FASTQ/pass/*.fastq > $FASTQ/pass/pass.fastq

# add basecalled results to fast5
tombo preprocess annotate_raw_with_fastqs --fast5-basedir $FAST5_S --fastq-filenames $FASTQ/pass/pass.fastq \
--overwrite --sequencing-summary-filenames $FASTQ/sequencing_summary.txt

# resquigglie
tombo resquiggle $FAST5_S $REF --fit-global-scale --include-event-stdev --overwrite
tombo detect_modifications de_novo --fast5-basedirs $FAST5_S --statistics-file-basename $OUT/$JOB

tombo text_output browser_files \
--fast5-basedirs $FAST5_S \
--statistics-filename $OUT/$JOB.tombo.stats \
--browser-file-basename $OUT/$JOB \
--genome-fasta $REF \
--motif-descriptions DRACH:3:m6A \
--file-types coverage dampened_fraction fraction

wig2bed < $OUT/$JOB.dampened_fraction_modified_reads.m6A.plus.wig > $OUT/pred.bed


### MINES
TOMBO=/PATH/TO/TOMBO/OUTPUT/DIR
OUT=/PATH/TO/MINES/OUTPUT/DIR
MINES=/PATH/TO/MINES
REF=/PATH/TO/REF

tombo text_output browser_files \
--fast5-basedirs $TOMBO/fast5 \
--statistics-filename $TOMBO/$JOB.tombo.stats \
--browser-file-basename $TOMBO/mines \
--file-types coverage dampened_fraction fraction

awk '{if($0!=null){print $0}}' $TOMBO/mines.fraction_modified_reads.plus.wig > $OUT/mines.wig
wig2bed < $OUT/mines.wig > $OUT/mines.wig.bed

# replace 'cDNA_MINES.py' with 'cDNA_MINES_revised.py'
python $MINES/cDNA_MINES_revised.py --fraction_modified $OUT/mines.wig.bed \
--coverage $TOMBO/mines.coverage.plus.bedgraph \
--output $OUT/pred.bed \
--ref $REF \
--kmer_models $MINES/Final_Models/names.txt


### EpiNano
OUT=/PATH/TO/OUTPUT/DIR
FASTQDIR=$OUT/fastq
EPINANO=/PATH/TO/EPINANO
SAM2TSV=/PATH/TO/sam2tsv.jar
PICARD=/PATH/TO/picard.jar
FAST5=/PATH/TO/FAST5
JOB=epinano

# Basecalling by guppy v3.1.5
guppy_basecaller -c rna_r9.4.1_70bps_hac.cfg -i $FAST5 -s $FASTQDIR -r --compress_fastq
cat $FASTQDIR/*.fastq.gz > $FASTQDIR/pass.fastq.gz
FASTQ=$FASTQDIR/pass.fastq.gz

# Mapping
minimap2 --MD -ax map-ont $REF $FASTQ > $OUT/$JOB.sam
samtools view -hbS $OUT/$JOB.sam > $OUT/$JOB.bam
samtools sort $OUT/$JOB.bam -o $OUT/${JOB}_sorted.bam
samtools index $OUT/${JOB}_sorted.bam
BAM=$OUT/${JOB}_sorted.bam

# picard
java -jar $PICARD CreateSequenceDictionary R=$REF

# EpiNano
python $EPINANO/Epinano_Variants.py -R $REF -b $BAM -s $SAM2TSV --type t 
python $EPINANO/misc/Slide_Variants.py ${JOB}_sorted.plus_strand.per.site.csv 5 

python $EPINANO/Epinano_Predict.py \
--model $EPINANO/models/rrach.deltaQ3.deltaMis3.deltaDel3.linear.dump \
--predict ${JOB}_sorted.plus_strand.per.site.5mer.csv \
--columns 8,13,23 \
--out_prefix pred