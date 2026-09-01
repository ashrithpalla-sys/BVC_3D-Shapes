.PHONY: install data train-ae train-vae evaluate demo test

install:
	python3 -m pip install -e ".[dev]"

data:
	python3 -m stoolgen generate --config configs/quick.yaml

train-ae:
	python3 -m stoolgen train --config configs/quick.yaml --model ae

train-vae:
	python3 -m stoolgen train --config configs/quick.yaml --model vae

evaluate:
	python3 -m stoolgen evaluate --config configs/quick.yaml --model vae

demo:
	python3 -m stoolgen demo --config configs/quick.yaml

test:
	python3 -m pytest

