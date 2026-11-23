# Usage:  make setup   (once)  ->  make pipeline  ->  make test
PY := .venv/bin/python

.PHONY: setup pipeline test report notebook clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

pipeline:
	$(PY) src/stage1_data_prep.py > outputs/stage1_log.txt
	$(PY) src/stage2_distances_method1.py > outputs/stage2_log.txt
	$(PY) src/stage3_method2.py > outputs/stage3_log.txt
	$(PY) src/school_matching.py > outputs/matching_log.txt
	$(PY) src/stage4_validation.py > outputs/stage4_log.txt
	$(PY) src/stage5_fairness.py > outputs/stage5_log.txt
	$(PY) src/stage6_sensitivity.py > outputs/stage6_log.txt
	$(MAKE) report

report:
	$(PY) src/stage7_report.py > outputs/stage7_log.txt
	$(PY) src/export_docx.py

notebook:
	$(PY) src/build_notebook.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -f outputs/*.npy
