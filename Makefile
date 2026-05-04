.PHONY: install test eval run clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	python main.py

eval:
	python main.py --eval

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
