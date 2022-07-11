.PHONY: tests tests_all

tests:
	pytest -v -s --timeout=20 -m "not CI"

tests_all:
	pytest -v -s --timeout=20 -m "not local_only"
