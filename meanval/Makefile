SHELL := /bin/bash
.SECONDARY:

wmt15.all-en.csv: ../data/wmt15/wmt15-master/data
	cat $</wmt15.*-eng.csv | grep -v srcIndex > $@.tmp
	cat <(head -1 $</wmt15.ces-eng.csv) $@.tmp > $@
	rm $@.tmp
