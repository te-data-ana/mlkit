# Mushroom data

The mushroom pipelines create local, generated data:

- `raw/uci/`: UCI feature and target snapshots, metadata and variable details
- `prepared.parquet`: validated features and the binary target
- `splits/`: stratified training, validation, test and monitoring cohorts
- `manifests/` and `predictions/`: training manifests and batch predictions
- `api_logs/`, `api_predictions/`, `api_errors/` and `monitoring/`: simulated
  serving logs and monitoring datasets and reports

These outputs are not stored in Git. The dataset contains simulated mushrooms;
models trained on it must not be used to assess whether mushrooms are safe to eat.
