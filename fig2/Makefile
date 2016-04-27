DATADIR = data
RM = /bin/rm -f

agg: aggtauenvcut aggpienvcut aggphases

aggphases:
	python ../aggregate.py $(DATADIR)/scan_phases phases

aggtauenvcut:
	python ../aggregate.py $(DATADIR)/scan_tauenvcut tauenvcut

aggpienvcut:
	python ../aggregate.py $(DATADIR)/scan_pienvcut pienvcut

run: runtauenvcut runpienvcut runphases

runphases:
	qsub submit_phases.sh

runtauenvcut:
	qsub submit_tauenvcut.sh

runpienvcut:
	qsub submit_pienvcut.sh

clean:
	$(RM) data/scan*

