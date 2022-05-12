
run:
	python3 main.py

mypy:
	mypy --strict .

.PHONY: run mypy
