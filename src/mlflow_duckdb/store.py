import os
import time
import uuid
import duckdb

from mlflow.entities import Experiment, ExperimentTag
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, RESOURCE_ALREADY_EXISTS
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.file_store import ViewType
from mlflow.store.entities.paged_list import PagedList
from mlflow.entities import Run, RunData, RunInfo, RunTag
from mlflow.utils.time import get_current_time_millis
from mlflow.entities.run_status import RunStatus
from mlflow.entities import Metric, Param
from mlflow.utils.search_utils import SearchUtils
from mlflow.entities.run_info import check_run_is_active
from mlflow.entities.run_inputs import RunInputs
from mlflow.entities.input_tag import InputTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.dataset import Dataset
from mlflow.store.tracking.file_store import _DatasetSummary
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
)
from mlflow.entities.run_info import RunInfo


class DuckDBTrackingStore(AbstractStore):
    DEFAULT_EXPERIMENT_ID = "0"

    def __init__(self, store_uri: str, artifact_uri: str | None = None):
        super().__init__()

        default_path = os.path.expanduser("~/mlruns.duckdb")

        # Parse the DuckDB URI and resolve a safe database path
        if store_uri.startswith("duckdb:///"):
            raw_path = store_uri[len("duckdb:///"):].strip()
            if not raw_path:  # Empty path after duckdb:///
                self.db_path = default_path
            elif os.path.isabs(raw_path):
                self.db_path = raw_path
            else:
                self.db_path = os.path.abspath(raw_path)
        elif store_uri.startswith("duckdb://"):
            # Handle duckdb:// (with two slashes) - extract everything after
            raw_path = store_uri[len("duckdb://"):].strip()
            if not raw_path:  # Empty path after duckdb://
                self.db_path = default_path
            elif os.path.isabs(raw_path):
                self.db_path = raw_path
            else:
                self.db_path = os.path.abspath(raw_path)
        elif store_uri.startswith("duckdb:"):
            # Handle duckdb: (with one colon) - extract everything after
            raw_path = store_uri[len("duckdb:"):].strip()
            if not raw_path:  # Empty path after duckdb:
                self.db_path = default_path
            elif os.path.isabs(raw_path):
                self.db_path = raw_path
            else:
                self.db_path = os.path.abspath(raw_path)
        else:
            msg = f"Invalid DuckDB store URI: {store_uri}. Must start with 'duckdb://', 'duckdb:///', or 'duckdb:'."
            raise MlflowException(
                msg,
            )

        # Ensure the directory for the DuckDB file exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Determine read-only mode from env
        read_only = os.environ.get("MLFLOW_DUCKDB_READ_ONLY", "0") == "1"

        if read_only and not os.path.exists(self.db_path):
            msg = (
                f"Cannot open database '{self.db_path}' in read-only mode: database does not exist. "
                "Ensure the database file exists before launching MLflow UI in read-only mode."
            )
            raise MlflowException(
                msg,
            )

        # Establish connection and initialize schema if not read-only
        self.conn = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._setup_schema()
            self._ensure_default_experiment()

    def _setup_schema(self):
        # Table: experiments
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                artifact_location TEXT,
                lifecycle_stage TEXT,
                creation_time BIGINT,
                last_update_time BIGINT
            )
        """)

        # Table: experiment tags
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_tags (
                experiment_id TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (experiment_id, key)
            )
        """)

        # Table: runs (run metadata)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                name TEXT,
                user_id TEXT,
                status TEXT,
                start_time BIGINT,
                end_time BIGINT,
                artifact_uri TEXT,
                lifecycle_stage TEXT,
                creation_time BIGINT,
                last_update_time BIGINT
            )
        """)

        # Table: metrics (multiple entries per key per run — full history)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT,
                key TEXT,
                value DOUBLE,
                timestamp BIGINT,
                step BIGINT,
                PRIMARY KEY (run_id, key, timestamp, step)
            )
        """)

        # Table: params (one param per key per run)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS params (
                run_id TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (run_id, key)
            )
        """)

        # Table: tags (one tag per key per run)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                run_id TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (run_id, key)
            )
        """)

        self.conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    run_id TEXT,
                    name TEXT,
                    digest TEXT,
                    source_type TEXT,
                    source TEXT,
                    schema TEXT,
                    tags TEXT
                )
            """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                request_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                timestamp_ms BIGINT,
                execution_time_ms BIGINT,
                status TEXT,
                request_metadata TEXT,
                tags TEXT
            )
        """)

    def _ensure_default_experiment(self):
        exists = self.conn.execute("""
            SELECT COUNT(*) FROM experiments WHERE experiment_id = ?
        """, (self.DEFAULT_EXPERIMENT_ID,)).fetchone()[0]
        if not exists:
            timestamp = int(time.time() * 1000)
            self.conn.execute("""
                INSERT INTO experiments (
                    experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.DEFAULT_EXPERIMENT_ID,
                "Default",
                f"{self.db_path}/artifacts/{self.DEFAULT_EXPERIMENT_ID}",
                LifecycleStage.ACTIVE,
                timestamp,
                timestamp,
            ))

    def create_experiment(self, name: str, artifact_location: str | None = None, tags: list[ExperimentTag] | None = None) -> str:
        existing = self.get_experiment_by_name(name)
        if existing:
            msg = f"Experiment '{name}' already exists."
            raise MlflowException(msg, RESOURCE_ALREADY_EXISTS)

        experiment_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        artifact_location = artifact_location or f"{self.db_path}/artifacts/{experiment_id}"

        self.conn.execute("""
            INSERT INTO experiments (
                experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (experiment_id, name, artifact_location, LifecycleStage.ACTIVE, timestamp, timestamp))

        return experiment_id

    def get_experiment(self, experiment_id: str) -> Experiment:
        result = self.conn.execute("""
            SELECT * FROM experiments WHERE experiment_id = ?
        """, (experiment_id,)).fetchone()

        if result is None:
            msg = f"Experiment ID {experiment_id} not found"
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        tags_list = self.get_all_experiment_tags(experiment_id)
        return Experiment(
            experiment_id=result[0],
            name=result[1],
            artifact_location=result[2],
            lifecycle_stage=result[3],
            tags=tags_list,
        )

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        result = self.conn.execute("""
            SELECT * FROM experiments WHERE name = ?
        """, (name,)).fetchone()

        if result is None:
            return None

        tags_list = self.get_all_experiment_tags(result[0])
        return Experiment(
            experiment_id=result[0],
            name=result[1],
            artifact_location=result[2],
            lifecycle_stage=result[3],
            tags=tags_list,
        )

    def delete_experiment(self, experiment_id: str) -> None:
        now = int(time.time() * 1000)
        self.conn.execute("""
            UPDATE experiments SET lifecycle_stage = ?, last_update_time = ? WHERE experiment_id = ?
        """, (LifecycleStage.DELETED, now, experiment_id))

    def restore_experiment(self, experiment_id: str) -> None:
        now = int(time.time() * 1000)
        self.conn.execute("""
            UPDATE experiments SET lifecycle_stage = ?, last_update_time = ? WHERE experiment_id = ?
        """, (LifecycleStage.ACTIVE, now, experiment_id))

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        now = int(time.time() * 1000)
        self.conn.execute("""
            UPDATE experiments SET name = ?, last_update_time = ? WHERE experiment_id = ?
        """, (new_name, now, experiment_id))

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        experiments, _ = self._search_experiments(view_type, 1000, None, None, None)
        return experiments

    def search_experiments(self, view_type=ViewType.ACTIVE_ONLY, max_results=100, filter_string=None, order_by=None, page_token=None):
        experiments, next_page_token = self._search_experiments(view_type, max_results, filter_string, order_by, page_token)
        return PagedList(experiments, next_page_token)

    def _search_experiments(self, view_type, max_results, filter_string=None, order_by=None, page_token=None) -> tuple[list[Experiment], str | None]:
        stages = LifecycleStage.view_type_to_stages(view_type)
        if not stages:
            return [], None

        placeholders = ','.join(['?'] * len(stages))
        sql = f"""
            SELECT experiment_id, name, artifact_location, lifecycle_stage FROM experiments
            WHERE lifecycle_stage IN ({placeholders})
            ORDER BY creation_time DESC
            LIMIT ?
        """
        params = [*list(stages), max_results]
        rows = self.conn.execute(sql, params).fetchall()

        experiments = []
        for row in rows:
            experiment_id = row[0]
            tags_list = self.get_all_experiment_tags(experiment_id)
            experiments.append(
                Experiment(
                    experiment_id=experiment_id,
                    name=row[1],
                    artifact_location=row[2],
                    lifecycle_stage=row[3],
                    tags=tags_list,
                ),
            )

        return experiments, None

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag):
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot set tag on experiment {experiment_id} because it is not active."
            raise MlflowException(msg)

        self.conn.execute("""
            INSERT INTO experiment_tags (experiment_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (experiment_id, key) DO UPDATE SET value = excluded.value
        """, (experiment_id, tag.key, tag.value))

    def get_all_experiment_tags(self, experiment_id: str) -> list[ExperimentTag]:
        rows = self.conn.execute("""
            SELECT key, value FROM experiment_tags WHERE experiment_id = ?
        """, (experiment_id,)).fetchall()
        return [ExperimentTag(key=row[0], value=row[1]) for row in rows]

    def create_run(self, experiment_id, user_id, start_time, tags, run_name=None):
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot create run in non-active experiment {experiment_id}"
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        run_id = uuid.uuid4().hex
        now = get_current_time_millis()
        artifact_uri = f"{experiment.artifact_location}/{run_id}"

        if run_name:
            tags = [*tags, RunTag(key="mlflow.runName", value=run_name)]

        self.conn.execute("""
            INSERT INTO runs (
                run_id, experiment_id, name, user_id, status,
                start_time, end_time, artifact_uri, lifecycle_stage,
                creation_time, last_update_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            experiment_id,
            "",  # still stored as tag
            user_id,
            RunStatus.to_string(RunStatus.RUNNING),
            start_time,
            None,
            artifact_uri,
            LifecycleStage.ACTIVE,
            now,
            now,
        ))

        for tag in tags:
            self.set_tag(run_id, tag)

        return self.get_run(run_id)

    def _get_run_info(self, run_id: str) -> RunInfo:
        row = self.conn.execute("""
            SELECT run_id, experiment_id, user_id, status,
                start_time, end_time, artifact_uri,
                lifecycle_stage, creation_time, last_update_time
            FROM runs WHERE run_id = ?
        """, (run_id,)).fetchone()

        if row is None:
            msg = f"Run '{run_id}' not found"
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        run_info = RunInfo(
            run_uuid=row[0],
            run_id=row[0],  # backward compatibility
            experiment_id=row[1],
            user_id=row[2],
            status=row[3],
            start_time=row[4],
            end_time=row[5],
            artifact_uri=row[6],
            lifecycle_stage=row[7],
        )

        # Store extra metadata for internal use, debugging, or compatibility
        run_info._creation_time = row[8]
        run_info._last_update_time = row[9]

        return run_info

    def get_run(self, run_id):
        run_info = self._get_run_info(run_id)
        if run_info is None:
            msg = f"Run '{run_id}' not found"
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        metrics = self.get_all_metrics(run_id)
        params = self.get_all_params(run_id)
        tags = self.get_all_tags(run_id)

        run_data = RunData(metrics=metrics, params=params, tags=tags)
        return Run(run_info=run_info, run_data=run_data, run_inputs=self._get_all_inputs(run_id))

    def get_all_metrics(self, run_id: str) -> list[Metric]:
        # Select the latest metric per (run_id, key)
        rows = self.conn.execute("""
            SELECT key, value, timestamp, step
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY key
                        ORDER BY timestamp DESC, step DESC
                    ) as row_num
                FROM metrics
                WHERE run_id = ?
            )
            WHERE row_num = 1
        """, (run_id,)).fetchall()

        return [Metric(key=row[0], value=row[1], timestamp=row[2], step=row[3]) for row in rows]

    def get_all_params(self, run_id: str) -> list[Param]:
        rows = self.conn.execute("""
            SELECT key, value FROM params WHERE run_id = ?
        """, (run_id,)).fetchall()

        return [Param(key=row[0], value=row[1]) for row in rows]

    def get_all_tags(self, run_id: str) -> list[RunTag]:
        rows = self.conn.execute("""
            SELECT key, value FROM tags WHERE run_id = ?
        """, (run_id,)).fetchall()

        return [RunTag(key=row[0], value=row[1]) for row in rows]

    def set_tag(self, run_id: str, tag: RunTag):
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot set tag on deleted run '{run_id}'"
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        self.conn.execute("""
            INSERT INTO tags (run_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id, key) DO UPDATE SET value = excluded.value
        """, (run_id, tag.key, tag.value))

    def log_param(self, run_id: str, param: Param):
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot log param to deleted run '{run_id}'"
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        self.conn.execute("""
            INSERT INTO params (run_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id, key) DO NOTHING
        """, (run_id, param.key, param.value))

    def log_metric(self, run_id: str, metric: Metric):
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot log metric to deleted run '{run_id}'"
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        self.conn.execute("""
            INSERT INTO metrics (run_id, key, value, timestamp, step)
            VALUES (?, ?, ?, ?, ?)
        """, (run_id, metric.key, metric.value, metric.timestamp, metric.step))

    def get_metric_history(
        self, run_id: str, metric_key: str, max_results: int | None = None, page_token: str | None = None,
    ) -> PagedList:
        query = """
            SELECT key, value, timestamp, step
            FROM metrics
            WHERE run_id = ? AND key = ?
            ORDER BY timestamp, step
        """
        params = [run_id, metric_key]

        if max_results is not None:
            query += " LIMIT ?"
            params.append(max_results)

        rows = self.conn.execute(query, params).fetchall()

        metrics = [Metric(key=row[0], value=row[1], timestamp=row[2], step=row[3]) for row in rows]
        return PagedList(metrics, token=None)

    def _list_run_infos(self, experiment_id: str, view_type: ViewType) -> list[RunInfo]:
        stages = LifecycleStage.view_type_to_stages(view_type)
        placeholders = ",".join(["?"] * len(stages))

        sql = f"""
            SELECT run_id, experiment_id, user_id, status, start_time, end_time,
                artifact_uri, lifecycle_stage
            FROM runs
            WHERE experiment_id = ? AND lifecycle_stage IN ({placeholders})
            ORDER BY start_time DESC
        """
        params = [experiment_id, *list(stages)]
        rows = self.conn.execute(sql, params).fetchall()

        return [
            RunInfo(
                run_uuid=row[0],
                run_id=row[0],
                experiment_id=row[1],
                user_id=row[2],
                status=row[3],
                start_time=row[4],
                end_time=row[5],
                artifact_uri=row[6],
                lifecycle_stage=row[7],
            )
            for row in rows
        ]

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str | None,
        run_view_type: ViewType,
        max_results: int,
        order_by: list[str] | None,
        page_token: str | None,
    ) -> PagedList:
        runs = []
        for experiment_id in experiment_ids:
            run_infos = self._list_run_infos(experiment_id, run_view_type)
            for run_info in run_infos:
                run = self.get_run(run_info.run_id)
                runs.append(run)

        # Apply filtering and sorting (in-memory for now)
        filtered = SearchUtils.filter(runs, filter_string)
        sorted_runs = SearchUtils.sort(filtered, order_by)
        paginated, next_token = SearchUtils.paginate(sorted_runs, page_token, max_results)

        return PagedList(paginated, token=next_token)

    def update_run_info(self, run_id: str, run_status: str, end_time: int, run_name: str) -> RunInfo:
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)

        new_info = run_info._copy_with_overrides(
            status=run_status,
            end_time=end_time,
            run_name=run_name,
        )

        self._overwrite_run_info(new_info)
        return new_info

    def delete_run(self, run_id: str):
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Run '{run_id}' is already deleted."
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        new_info = run_info._copy_with_overrides(
            lifecycle_stage=LifecycleStage.DELETED,
        )

        self._overwrite_run_info(new_info, deleted_time=get_current_time_millis())

    def restore_run(self, run_id: str):
        run_info = self._get_run_info(run_id)

        if run_info.lifecycle_stage != LifecycleStage.DELETED:
            msg = f"Run '{run_id}' is not deleted and cannot be restored."
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        new_info = run_info._copy_with_overrides(
            lifecycle_stage=LifecycleStage.ACTIVE,
        )

        self._overwrite_run_info(new_info, deleted_time=get_current_time_millis())

    def _overwrite_run_info(self, run_info: RunInfo, deleted_time: int | None = None):
        self.conn.execute("""
            UPDATE runs
            SET status = ?, end_time = ?, lifecycle_stage = ?, last_update_time = ?
            WHERE run_id = ?
        """, (
            run_info.status,
            run_info.end_time,
            run_info.lifecycle_stage,
            deleted_time or get_current_time_millis(),
            run_info.run_id,
        ))

    def log_inputs(self, run_id: str, datasets: list[DatasetInput]) -> None:
        for dataset_input in datasets:
            dataset = dataset_input.dataset
            self.conn.execute(
                """
                INSERT INTO datasets (
                    run_id, name, digest, source_type,
                    source, schema, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset.name,
                    dataset.digest,
                    dataset.source_type,
                    dataset.source,
                    dataset.schema,
                    ",".join(f"{t.key}:{t.value}" for t in dataset_input.tags or []),
                ),
            )

    def _get_all_inputs(self, run_id: str) -> RunInputs:
        rows = self.conn.execute("""
            SELECT name, digest, source_type, source, schema, tags
            FROM datasets WHERE run_id = ?
        """, (run_id,)).fetchall()

        dataset_inputs = []
        for row in rows:
            tags_str = row[5]  # now correct index
            tags = []
            if tags_str:
                for t in tags_str.split(","):
                    if ":" in t:
                        k, v = t.split(":", 1)
                        tags.append(InputTag(k, v))

            dataset = Dataset(
                name=row[0],
                digest=row[1],
                source_type=row[2],
                source=row[3],
                schema=row[4],
            )

            dataset_input = DatasetInput(dataset=dataset, tags=tags)
            dataset_inputs.append(dataset_input)

        return RunInputs(dataset_inputs=dataset_inputs)

    def _search_datasets(self, experiment_ids: list[str]) -> list[_DatasetSummary]:
        """Return all dataset summaries associated with the given experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.

        Returns:
            A List of :py:class:`mlflow.store.tracking.file_store._DatasetSummary` entities.

        """
        MAX_DATASET_SUMMARIES_RESULTS = 1000
        seen = set()
        summaries = []

        placeholders = ",".join(["?"] * len(experiment_ids))
        query = f"""
            SELECT r.experiment_id, d.name, d.digest, d.tags
            FROM datasets d
            JOIN runs r ON d.run_id = r.run_id
            WHERE r.experiment_id IN ({placeholders})
        """
        rows = self.conn.execute(query, experiment_ids).fetchall()

        for experiment_id, name, digest, tags_str in rows:
            context = None
            if tags_str:
                for tag in tags_str.split(","):
                    if ":" in tag:
                        k, v = tag.split(":", 1)
                        if k == MLFLOW_DATASET_CONTEXT:
                            context = v
                            break

            key = (experiment_id, name, digest, context)
            if key not in seen:
                seen.add(key)
                summaries.append(_DatasetSummary(
                    experiment_id=str(experiment_id),
                    name=name,
                    digest=digest,
                    context=context,
                ))

            if len(summaries) >= MAX_DATASET_SUMMARIES_RESULTS:
                break

        return summaries
