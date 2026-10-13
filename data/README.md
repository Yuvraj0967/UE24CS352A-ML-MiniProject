# Data

Nothing needs to be downloaded by hand. The first run of `python train_model.py` fetches the
[CONDA](https://github.com/usydnlp/CONDA) Dota 2 chat dataset into `data/raw/` and writes the cleaned
`train.csv` / `test.csv` into `data/processed/`. GloVe vectors (~70 MB) are fetched into `data/glove/`.
These folders are git-ignored.
