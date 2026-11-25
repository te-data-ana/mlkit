# Diabetes data

The diabetes pipelines create the following local, generated data:

- `raw/uci/`: UCI identifier, feature and target snapshots, metadata and variable details
- `prepared.parquet`: validated features, identifier metadata and binary target
- `splits/`: stratified training, validation, test and monitoring cohorts
- `manifests/` and `predictions/`: training manifests and batch predictions
- `api_logs/`, `api_predictions/`, `api_errors/` and `monitoring/`: simulated
  serving logs and monitoring datasets and reports

These outputs are not stored in Git. They contain health and demographic
information; restrict access and protect local copies appropriately.
