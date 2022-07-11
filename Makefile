.PHONY: tests tests_all

tests:
	pytest -v -s --timeout=10 -m "not CI"

tests_all:
	pytest -v -s --timeout=10
