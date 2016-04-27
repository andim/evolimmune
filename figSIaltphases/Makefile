DATADIR = data
RM = /bin/rm -f

agg: aggphases

aggphases:
	python ../aggregate.py $(DATADIR)/scan_phases phases

run: runphases

runphases:
	qsub submit_phases.sh

clean:
	$(RM) data/scan*

