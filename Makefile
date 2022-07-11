.PHONY: tests tests_all

tests:
	pytest -v -s --timeout=20 -m "not CI"

tests_all:
	pytest -v -s --timeout=20 howl/data/dataset/dataset_test.py howl/data/dataset/dataset_writer_test.py howl/dataset/raw_audio_dataset_generator_test.py
