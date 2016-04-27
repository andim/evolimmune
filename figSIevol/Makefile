DATADIR = data
RM = /bin/rm -f

agg: 
	python ../aggregate.py $(DATADIR)/scan_ind scan

run:
	qsub submit.sh

clean:
	$(RM) $(DATADIR)/scan*

