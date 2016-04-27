DATADIR = data
RM = /bin/rm -f

agg: 
	python ../aggregate.py $(DATADIR)/scan_opt opt

run:
	qsub submit.sh

clean:
	$(RM) data/scan*

