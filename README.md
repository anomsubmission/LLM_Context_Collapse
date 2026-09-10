# Running the code

Install the package:

```bash
python -m pip install -e .
```

Install the optional experiment dependencies:

```bash
python -m pip install -e ".[experiments]"
```

Run the environment check:

```bash
cliff doctor
```

Verify the archived results:

```bash
cliff verify
```

Regenerate figures:

```bash
cliff figures
```

Run the offline smoke test:

```bash
cliff sweep --backend echo --model echo --ks 0,1,4 --seeds 0 --n-test 20
```

Run all tests:

```bash
python -m pytest
```

Run an experiment:

```bash
cliff sweep --model meta-llama/Llama-3.1-8B-Instruct
cliff intervene --model meta-llama/Llama-3.1-8B-Instruct --arm remove --region mid
cliff mechanism --model meta-llama/Llama-3.1-8B-Instruct --task ag_news --k 16
cliff audit --model google/gemma-3-4b-it
```

Set output locations with `CLIFF_RESULTS`, `CLIFF_FIGURES`, and `CLIFF_OUTPUTS`.
