# Running the code

## Install

```bash
python -m pip install -e .
```

For experiment commands:

```bash
python -m pip install -e ".[experiments]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

## Verify archived results

```bash
cliff verify
```

Use a different results directory:

```bash
CLIFF_RESULTS=/path/to/results cliff verify
```

## Regenerate figures

```bash
cliff figures
```

Regenerate one figure:

```bash
cliff figures --only 4
```

## Run the smoke test

```bash
cliff sweep --backend echo --model echo --tasks ag_news --ks 0,1,4 --seeds 0 --n-test 20
```

## Run experiments

```bash
cliff sweep --model meta-llama/Llama-3.1-8B-Instruct
cliff intervene --model meta-llama/Llama-3.1-8B-Instruct --arm remove --region mid
cliff mechanism --model meta-llama/Llama-3.1-8B-Instruct --task ag_news --k 16
cliff audit --model google/gemma-3-4b-it
```

## Run tests

```bash
python -m pytest
```

## Make targets

```bash
make install
make install-all
make verify
make figures
make reproduce
make test
make smoke
```
