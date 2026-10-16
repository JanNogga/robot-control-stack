PYSRC = src/rcs_elrobot
TESTSRC = tests
PYTHONPATH_LOCAL = src

pycheckformat:
	isort --check-only ${PYSRC} ${TESTSRC}
	black --check ${PYSRC} ${TESTSRC}

pyformat:
	isort ${PYSRC} ${TESTSRC}
	black ${PYSRC} ${TESTSRC}

ruff:
	ruff check ${PYSRC} ${TESTSRC}

pytest:
	PYTHONPATH=${PYTHONPATH_LOCAL} python -m pytest ${TESTSRC}

test-compile:
	python -m compileall ${PYSRC} ${TESTSRC}

.PHONY: pycheckformat pyformat ruff pytest test-compile
