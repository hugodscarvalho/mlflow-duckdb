"""Generate synthetic experiments, runs, and dataset inputs for MLflow DuckDB tracking store."""

import mlflow
import time
import random
from mlflow.entities.dataset import Dataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag


def random_string():
    """Generate a random 8-character alphanumeric string."""
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))


def random_tags():
    """Generate random input tags for a dataset."""
    return [
        InputTag("source", random.choice(["auto", "manual"])),
        InputTag("context", "training"),
    ]


def random_dataset(exp_idx, run_idx):
    """Create a random DatasetInput object for testing purposes."""
    return DatasetInput(
        dataset=Dataset(
            name=f"dataset_exp{exp_idx}_run{run_idx}.csv",
            digest=f"sha256:{random.getrandbits(128):032x}",
            source_type=random.choice(["local", "s3", "dbfs"]),
            source=f"/data/exp{exp_idx}/run{run_idx}/dataset.csv",
            schema=f"schema_v{random.randint(1, 3)}",
        ),
        tags=random_tags(),
    )


def random_params():
    """Generate random ML model parameters."""
    return {
        "model_type": random.choice(["xgboost", "random_forest", "svm"]),
        "max_depth": str(random.randint(2, 10)),
        "learning_rate": str(round(random.uniform(0.01, 0.3), 3)),
    }


def random_metrics():
    """Generate random evaluation metrics."""
    return {
        "accuracy": round(random.uniform(0.7, 0.99), 3),
        "precision": round(random.uniform(0.6, 0.95), 3),
        "recall": round(random.uniform(0.5, 0.9), 3),
    }


mlflow.set_tracking_uri("duckdb://mlruns.duckdb")
client = mlflow.MlflowClient()

num_experiments = 3
runs_per_experiment = 5

for exp_idx in range(num_experiments):
    experiment_name = f"Experiment_{exp_idx}"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    for run_idx in range(runs_per_experiment):
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            run_id = run.info.run_id

            # Log Params
            for key, value in random_params().items():
                mlflow.log_param(key, value)

            # Log Tags
            mlflow.set_tag("user", f"user_{random.randint(1, 10)}")
            mlflow.set_tag("stage", random.choice(["dev", "staging", "prod"]))

            # Log Metrics
            for metric_name, value in random_metrics().items():
                mlflow.log_metric(metric_name, value)
                time.sleep(0.001)

            # Log Dataset Input
            dataset_input = random_dataset(exp_idx, run_idx)
            client.log_inputs(run_id, datasets=[dataset_input])

            mlflow.end_run()
