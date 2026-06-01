# Selected results

This folder contains selected result summaries used in the final report.

The complete repeated-seed and shot-noise files can be regenerated with:

```bash
python scripts/run_repeated_seed_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --n-seeds 10
python scripts/run_shot_noise_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --shots 100 1000 10000 --repeats 30
```
