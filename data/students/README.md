# Students data

The students pipelines create local, generated data:

- `raw/uci/`: UCI feature and target snapshots, metadata and variable details
- `prepared.parquet`: validated features and the three-class target
- `splits/`: stratified fit, validation, test and monitoring cohorts
- `manifests/` and `predictions/`: training manifests and batch predictions
- `api_logs/`, `api_predictions/`, `api_errors/` and `monitoring/`: simulated
  serving logs and monitoring datasets and reports

These outputs are not stored in Git. Models are for workflow demonstration,
not automated decisions about individual students.
