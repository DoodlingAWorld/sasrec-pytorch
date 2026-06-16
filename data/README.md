# Data

Datasets are **downloaded on demand** and are **git-ignored** (never committed).

- `scripts/prepare_data.py` downloads MovieLens-1M into `data/raw/ml-1m/ratings.dat`.
- Primary source: GroupLens (`files.grouplens.org`).
- Fallback: a Hugging Face mirror of the original `ratings.dat`, used automatically when
  the primary host is unreachable (e.g. a sandbox where only `huggingface.co` is allowlisted).

Format (`ratings.dat`): `UserID::MovieID::Rating::Timestamp` (rating is ignored — we treat
any interaction as implicit positive feedback, per the paper).
