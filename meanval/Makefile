SHELL := /bin/bash

.SECONDARY:

### LINUX
SHUF=shuf

### OSX Only
# FIXME: Add if statement. Use uname or smt.
SHUF=gshuf


### COMMON ENVIRONMENT VARIABLES
PY_ENV=py3-tf  # change it with appropriate one

### CONSTANTS
RANK_IDX=2
REFERENCE_IDX=9
MT_IDX=10
DATASET_DIR=datasets


wmt15.all-en.csv: ../data/wmt15/wmt15-master/data
	cat $</wmt15.*-eng.csv | grep -v srcIndex > $@.tmp
	cat <(head -1 $</wmt15.ces-eng.csv) $@.tmp > $@
	rm $@.tmp

%.csv: %.all-en.csv
	source activate ${PY_ENV}
	python run_preprocess.py $<

prep-%: %.csv
	${SHUF} $< > ${DATASET_DIR}/$<.shuffled
	cat $< | head -49937 | cut -f ${RANK_IDX} > ${DATASET_DIR}/$*.train.labels.txt
	cat $< | head -49937 | cut -f ${REFERENCE_IDX} > ${DATASET_DIR}/$*.train.ref.sentences.txt
	cat $< | head -49937 | cut -f ${MT_IDX} > ${DATASET_DIR}/$*.train.mt.sentences.txt
	cat $< | tail -10000 | cut -f ${RANK_IDX} > ${DATASET_DIR}/$*.validate.labels.txt
	cat $< | tail -10000 | cut -f ${REFERENCE_IDX} > ${DATASET_DIR}/$*.validate.ref.sentences.txt
	cat $< | tail -10000 | cut -f ${MT_IDX} > ${DATASET_DIR}/$*.validate.mt.sentences.txt
	rm ${DATASET_DIR}/$<.shuffled

prep-all:
	-mkdir ${DATASET_DIR}
	make prep-wmt15
