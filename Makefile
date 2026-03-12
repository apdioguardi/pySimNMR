.PHONY: test

test:
	pytest -q --basetemp .pytest_tmp
