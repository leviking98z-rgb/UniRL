# Data

`unirl.data` owns runtime data sources, supervised-example normalization, and
offline dataset preparation. Preparation commands live under
`unirl.data.prepare`:

```bash
python -m unirl.data.prepare.dapo_math --out-dir data/dapo_math
python -m unirl.data.prepare.geo3k_mc --out-dir data/geo3k_mc
python -m unirl.data.prepare.sft_text --out-dir data/sft_alpaca
```

The former `python -m unirl.utils.prepare_*` commands remain as thin
compatibility entrypoints.
