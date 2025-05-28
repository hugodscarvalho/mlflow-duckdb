"""DuckDB-backed implementation of the MLflow Tracking Store.

This module implements all the required interfaces for tracking experiments,
runs, metrics, parameters, tags, and datasets using DuckDB as a backend.
"""

import uuid
import duckdb

# MLflow core entities and utilities
from mlflow.entities import (
    Experiment,
    ExperimentTag,
    Run,
    RunData,
    RunInfo,
    RunTag,
    Metric,
    Param,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_status import RunStatus
from mlflow.entities.run_info import check_run_is_active
from mlflow.entities.run_inputs import RunInputs
from mlflow.entities.input_tag import InputTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.dataset import Dataset

# MLflow tracking and protocol
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    RESOURCE_ALREADY_EXISTS,
)

# MLflow tracking store interfaces
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.file_store import ViewType, _DatasetSummary
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.time import get_current_time_millis

# DuckDB-specific configuration and setup
from mlflow_duckdb.config.constants import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_MAX_EXPERIMENT_RESULTS,
    EXPERIMENT_ARTIFACT_PATH_TEMPLATE,
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_RUN_NAME_TAG,
    ERR_MSG_RUN_NOT_FOUND,
    MAX_DATASET_SUMMARIES_RESULTS,
)
from mlflow_duckdb.config.env import is_read_only, validate_duckdb_access
from mlflow_duckdb.config.path_resolver import (
    resolve_duckdb_path,
    ensure_db_directory_exists,
    get_default_db_path,
)
from mlflow_duckdb.schema.schema_manager import setup_schema
from mlflow_duckdb.bootstrap.default_experiment import ensure_default_experiment


class DuckDBTrackingStore(AbstractStore):
    """MLflow Tracking Store backed by a DuckDB database."""

    def __init__(self, store_uri: str, artifact_uri: str | None = None):
        """Initialize a DuckDB-backed MLflow Tracking Store.

        Resolves the path to the DuckDB database, ensures the directory exists,
        checks read-only mode, and establishes a connection to the DuckDB file.
        If not in read-only mode, also initializes the schema and default experiment.

        Args:
            store_uri (str): URI that identifies the DuckDB store location.
            artifact_uri (str | None): Optional artifact root URI. Currently unused.

        Raises:
            MlflowException: If the URI is invalid or access is not permitted.

        """
        super().__init__()

        # Resolve DuckDB path from the URI
        self.db_path = resolve_duckdb_path(store_uri=store_uri,
                                           default_path=get_default_db_path(),
                                        )

        # Ensure the target folder exists
        ensure_db_directory_exists(self.db_path)

        # Check for read-only mode and validate access
        read_only = is_read_only()
        validate_duckdb_access(self.db_path, read_only)

        # Connect to the DuckDB database
        self.conn = duckdb.connect(self.db_path, read_only=read_only)

        # Bootstrap schema and default experiment if not read-only
        if not read_only:
            setup_schema(self.conn.cursor())
            ensure_default_experiment(cursor=self.conn.cursor(),
                                      db_path=self.db_path,
                                      default_id=DEFAULT_EXPERIMENT_ID,
                                    )

    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: list[ExperimentTag] | None = None,
    ) -> str:
        """Create a new MLflow experiment.

        This method generates a new UUID-based experiment ID, stores the metadata
        in the DuckDB `experiments` table, and returns the experiment ID. If an
        experiment with the same name already exists, an exception is raised.

        Args:
            name (str): Name of the new experiment.
            artifact_location (Optional[str]): Optional URI for the experiment's
                artifact location.
            tags (Optional[List[ExperimentTag]]): Tags associated with the
                experiment (currently unused).

        Returns:
            str: The generated experiment ID.

        Raises:
            MlflowException: If an experiment with the same name already exists.

        """
        existing = self.get_experiment_by_name(name)
        if existing:
            msg = f"Experiment '{name}' already exists."
            raise MlflowException(msg, RESOURCE_ALREADY_EXISTS)

        experiment_id = str(uuid.uuid4())
        timestamp = get_current_time_millis()

        artifact_location = (
            artifact_location
            or EXPERIMENT_ARTIFACT_PATH_TEMPLATE.format(
                db_path=self.db_path,
                experiment_id=experiment_id,
            )
        )

        self.conn.execute(
            """
            INSERT INTO experiments (
                experiment_id,
                name,
                artifact_location,
                lifecycle_stage,
                creation_time,
                last_update_time
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                name,
                artifact_location,
                LifecycleStage.ACTIVE,
                timestamp,
                timestamp,
            ),
        )

        return experiment_id

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve an MLflow experiment by its ID.

        Executes a query against the DuckDB `experiments` table to fetch
        experiment metadata. Raises an exception if the experiment does not exist.

        Args:
            experiment_id (str): The unique ID of the experiment to retrieve.

        Returns:
            Experiment: An MLflow Experiment object containing metadata and tags.

        Raises:
            MlflowException: If the experiment ID does not exist in the database.

        """
        result = self.conn.execute(
            """
            SELECT experiment_id, name, artifact_location, lifecycle_stage
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        if result is None:
            msg = f"Experiment ID {experiment_id} not found"
            raise MlflowException(
                msg,
                RESOURCE_DOES_NOT_EXIST,
            )

        # Retrieve tags associated with the experiment
        tags_list = self.get_all_experiment_tags(experiment_id)

        # Build and return the Experiment object
        return Experiment(
            experiment_id=result[0],
            name=result[1],
            artifact_location=result[2],
            lifecycle_stage=result[3],
            tags=tags_list,
        )

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        """Retrieve an MLflow experiment by its name.

        Looks up the `experiments` table for a given experiment name.
        Returns `None` if the experiment is not found.

        Args:
            name (str): The name of the experiment to retrieve.

        Returns:
            Optional[Experiment]: The matching MLflow Experiment, or None if not found.

        """
        result = self.conn.execute(
            """
            SELECT experiment_id, name, artifact_location, lifecycle_stage
            FROM experiments
            WHERE name = ?
            """,
            (name,),
        ).fetchone()

        if result is None:
            return None

        # Retrieve tags using the experiment ID
        tags_list = self.get_all_experiment_tags(result[0])

        return Experiment(
            experiment_id=result[0],
            name=result[1],
            artifact_location=result[2],
            lifecycle_stage=result[3],
            tags=tags_list,
        )

    def delete_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as deleted by updating its lifecycle stage.

        This performs a soft delete: the experiment remains in the database
        but is excluded from active views.

        Args:
            experiment_id (str): The unique ID of the experiment to delete.

        """
        now = get_current_time_millis()

        self.conn.execute(
            """
            UPDATE experiments
            SET lifecycle_stage = ?, last_update_time = ?
            WHERE experiment_id = ?
            """,
            (LifecycleStage.DELETED, now, experiment_id),
        )

    def restore_experiment(self, experiment_id: str) -> None:
        """Restore a deleted experiment by setting its lifecycle stage to active.

        Args:
            experiment_id (str): The unique identifier of the experiment to restore.

        """
        now = get_current_time_millis()

        self.conn.execute(
            """
            UPDATE experiments
            SET lifecycle_stage = ?, last_update_time = ?
            WHERE experiment_id = ?
            """,
            (LifecycleStage.ACTIVE, now, experiment_id),
        )

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        """Rename an existing experiment.

        Args:
            experiment_id (str): The unique identifier of the experiment to rename.
            new_name (str): The new name to assign to the experiment.

        """
        now = get_current_time_millis()

        self.conn.execute(
            """
            UPDATE experiments
            SET name = ?, last_update_time = ?
            WHERE experiment_id = ?
            """,
            (new_name, now, experiment_id),
        )

    def list_experiments(self,
                         view_type: ViewType = ViewType.ACTIVE_ONLY,
                        ) -> list[Experiment]:
        """List all experiments, optionally filtered by view type.

        Args:
            view_type (ViewType, optional): Type of experiments to view.
                Options include ACTIVE_ONLY, DELETED_ONLY, and ALL.
                Defaults to ACTIVE_ONLY.

        Returns:
            list[Experiment]: A list of experiments matching the view type.

        """
        experiments, _ = self._search_experiments(
            view_type=view_type,
            max_results=DEFAULT_MAX_EXPERIMENT_RESULTS,
            filter_string=None,
            order_by=None,
            page_token=None,
        )
        return experiments

    def search_experiments(
        self,
        view_type: ViewType = ViewType.ACTIVE_ONLY,
        max_results: int = 100,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList:
        """Search for experiments based on optional filters, ordering, and pagination.

        Args:
            view_type (ViewType): View type to filter by
                (e.g., ACTIVE_ONLY, DELETED_ONLY).
            max_results (int): Maximum number of results to return.
            filter_string (Optional[str]): Filter string to apply to experiments.
            order_by (Optional[List[str]]): Fields to order by.
            page_token (Optional[str]): Token for pagination.

        Returns:
            PagedList: A paginated list of Experiment objects and a
                next page token (if any).

        """
        experiments, next_page_token = self._search_experiments(
            view_type, max_results, filter_string, order_by, page_token,
        )

        return PagedList(items=experiments, token=next_page_token)

    def _search_experiments(
        self,
        view_type: ViewType,
        max_results: int,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> tuple[list[Experiment], str | None]:
        """Search for experiments by lifecycle stage and return a paginated result.

        Args:
            view_type (ViewType): Which experiment stages to include
                (active, deleted, or all).
            max_results (int): Maximum number of experiments to return.
            filter_string (Optional[str], optional): Not yet implemented.
            order_by (Optional[List[str]], optional): Not yet implemented.
            page_token (Optional[str], optional): Not yet implemented.

        Returns:
            Tuple[List[Experiment], Optional[str]]: A list of matching experiments and
                an optional next page token.

        """
        stages = LifecycleStage.view_type_to_stages(view_type)
        if not stages:
            return [], None

        # Prepare placeholders and query dynamically based on number of stages
        placeholders = ",".join(["?"] * len(stages))
        sql = f"""
            SELECT experiment_id, name, artifact_location, lifecycle_stage
            FROM experiments
            WHERE lifecycle_stage IN ({placeholders})
            ORDER BY creation_time DESC
            LIMIT ?
        """

        params = [*stages, max_results]
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

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        """Set or update a tag for an existing experiment.

        If the tag already exists for the given experiment, its value is updated.

        Args:
            experiment_id (str): The ID of the experiment to tag.
            tag (ExperimentTag): The tag to set or update.

        Raises:
            MlflowException: If the experiment is not in an active state.

        """
        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = (f"Cannot set tag on experiment {experiment_id} "
                   "because it is not active.")
            raise MlflowException(msg)

        # Insert new tag or update the value if the key already exists
        self.conn.execute(
            """
            INSERT INTO experiment_tags (experiment_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (experiment_id, key) DO UPDATE SET value = excluded.value
            """,
            (experiment_id, tag.key, tag.value),
        )

    def get_all_experiment_tags(self, experiment_id: str) -> list[ExperimentTag]:
        """Retrieve all tags associated with a given experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            list[ExperimentTag]: A list of all tags attached to the experiment.

        """
        rows = self.conn.execute(
            """
            SELECT key, value FROM experiment_tags WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchall()

        return [ExperimentTag(key=row[0], value=row[1]) for row in rows]

    def create_run(
        self,
        experiment_id: str,
        user_id: str,
        start_time: int,
        tags: list[RunTag],
        run_name: str | None = None,
    ) -> Run:
        """Create a new MLflow run under the specified experiment.

        Args:
            experiment_id (str): ID of the experiment the run belongs to.
            user_id (str): ID of the user who initiated the run.
            start_time (int): Unix timestamp (ms) of when the run started.
            tags (list[RunTag]): Tags associated with the run.
            run_name (Optional[str]): Optional name for the run. If provided,
                it will be added as a tag.

        Returns:
            Run: The created MLflow run object.

        Raises:
            MlflowException: If the experiment is not in an active lifecycle stage.

        """
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
            tags = [*tags, RunTag(key=MLFLOW_RUN_NAME_TAG, value=run_name)]

        self.conn.execute(
            """
            INSERT INTO runs (
                run_id, experiment_id, name, user_id, status,
                start_time, end_time, artifact_uri, lifecycle_stage,
                creation_time, last_update_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                experiment_id,
                "",  # name is stored as a tag
                user_id,
                RunStatus.to_string(RunStatus.RUNNING),
                start_time,
                None,
                artifact_uri,
                LifecycleStage.ACTIVE,
                now,
                now,
            ),
        )

        for tag in tags:
            self.set_tag(run_id, tag)

        return self.get_run(run_id)

    def _get_run_info(self, run_id: str) -> RunInfo:
        """Fetch the RunInfo object for the specified run ID.

        Args:
            run_id (str): The ID of the run to retrieve.

        Returns:
            RunInfo: The run metadata, including status, timing,
                and artifact location.

        Raises:
            MlflowException: If the run does not exist.

        """
        row = self.conn.execute(
            """
            SELECT run_id, experiment_id, user_id, status,
                start_time, end_time, artifact_uri,
                lifecycle_stage, creation_time, last_update_time
            FROM runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            raise MlflowException(ERR_MSG_RUN_NOT_FOUND.format(run_id=run_id),
                                  RESOURCE_DOES_NOT_EXIST,
                            )

        # Construct RunInfo using fetched data
        run_info = RunInfo(
            run_uuid=row[0],
            run_id=row[0],  # Kept for backward compatibility
            experiment_id=row[1],
            user_id=row[2],
            status=row[3],
            start_time=row[4],
            end_time=row[5],
            artifact_uri=row[6],
            lifecycle_stage=row[7],
        )

        # Attach internal-use metadata
        run_info._creation_time = row[8]
        run_info._last_update_time = row[9]

        return run_info

    def get_run(self, run_id: str) -> Run:
        """Retrieve a run and all its associated data.

        Args:
            run_id (str): The unique identifier of the run.

        Returns:
            Run: A complete MLflow Run object including metadata, metrics,
                parameters, tags, and inputs.

        Raises:
            MlflowException: If the run ID does not exist.

        """
        run_info = self._get_run_info(run_id)

        metrics = self.get_all_metrics(run_id)
        params = self.get_all_params(run_id)
        tags = self.get_all_tags(run_id)
        inputs = self._get_all_inputs(run_id)

        run_data = RunData(metrics=metrics, params=params, tags=tags)
        return Run(run_info=run_info, run_data=run_data, run_inputs=inputs)

    def get_all_metrics(self, run_id: str) -> list[Metric]:
        """Retrieve the latest value for each metric logged in a given run.

        This method uses a window function to return only the most recent value
        for each metric key, based on timestamp and step.

        Args:
            run_id (str): The unique ID of the run.

        Returns:
            list[Metric]: A list of Metric objects representing the latest
                values per key.

        """
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

        return [
            Metric(
                key=row[0],
                value=row[1],
                timestamp=row[2],
                step=row[3],
            ) for row in rows
        ]

    def get_all_params(self, run_id: str) -> list[Param]:
        """Fetch all parameters associated with a given run.

        Args:
            run_id (str): The unique ID of the run.

        Returns:
            list[Param]: A list of Param objects, each representing a parameter
                key-value pair.

        """
        rows = self.conn.execute("""
            SELECT key, value FROM params WHERE run_id = ?
        """, (run_id,)).fetchall()

        return [Param(key=row[0], value=row[1]) for row in rows]

    def get_all_tags(self, run_id: str) -> list[RunTag]:
        """Fetch all tags associated with a given run.

        Args:
            run_id (str): The unique ID of the run.

        Returns:
            list[RunTag]: A list of RunTag objects, each representing a tag
                key-value pair.

        """
        rows = self.conn.execute("""
            SELECT key, value FROM tags WHERE run_id = ?
        """, (run_id,)).fetchall()

        return [RunTag(key=row[0], value=row[1]) for row in rows]

    def set_tag(self, run_id: str, tag: RunTag) -> None:
        """Set or update a tag for the specified run.

        This method will insert a new tag or update the value if the
        tag already exists.

        Args:
            run_id (str): The ID of the run to which the tag should be associated.
            tag (RunTag): The tag object containing the key and value to set.

        Raises:
            MlflowException: If the run is not active or does not exist.

        """
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot set tag on deleted run '{run_id}'"
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        self.conn.execute("""
            INSERT INTO tags (run_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id, key) DO UPDATE SET value = excluded.value
        """, (run_id, tag.key, tag.value))

    def log_param(self, run_id: str, param: Param) -> None:
        """Log a parameter to the specified run.

        Parameters are immutable once set—attempts to overwrite an existing key will
        be ignored.

        Args:
            run_id (str): The ID of the run to associate the parameter with.
            param (Param): A parameter object containing the key and value to log.

        Raises:
            MlflowException: If the run is not active or does not exist.

        """
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot log param to deleted run '{run_id}'"
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        self.conn.execute("""
            INSERT INTO params (run_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id, key) DO NOTHING
        """, (run_id, param.key, param.value))

    def log_metric(self, run_id: str, metric: Metric) -> None:
        """Log a metric to the specified run.

        Unlike parameters, metrics are not immutable and multiple values for the same
        key can exist across different timestamps and steps.

        Args:
            run_id (str): The ID of the run to associate the metric with.
            metric (Metric): A metric object containing the key, value, timestamp,
                and step.

        Raises:
            MlflowException: If the run is not active or does not exist.

        """
        run_info = self._get_run_info(run_id)
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Cannot log metric to deleted run '{run_id}'"
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        self.conn.execute("""
            INSERT INTO metrics (run_id, key, value, timestamp, step)
            VALUES (?, ?, ?, ?, ?)
        """, (run_id, metric.key, metric.value, metric.timestamp, metric.step))

    def get_metric_history(
        self,
        run_id: str,
        metric_key: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList:
        """Retrieve the full history of a given metric for a specific run.

        Args:
            run_id (str): The ID of the run.
            metric_key (str): The name of the metric to retrieve.
            max_results (int | None, optional): Maximum number of metric entries
                to return.
                If None, all results are returned. Defaults to None.
            page_token (str | None, optional): Pagination token. Currently unused.
                Defaults to None.

        Returns:
            PagedList: A paginated list of Metric objects sorted by timestamp and step.

        """
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

        metrics = [
            Metric(key=row[0], value=row[1], timestamp=row[2], step=row[3])
            for row in rows
        ]
        return PagedList(metrics, token=None)

    def _list_run_infos(self, experiment_id: str, view_type: ViewType) -> list[RunInfo]:
        """Retrieve run metadata for a given experiment ID and view type.

        This method fetches basic run-level metadata (RunInfo) for all runs
        within the given experiment ID, filtered by the lifecycle stage
        associated with the specified `view_type`.

        Args:
            experiment_id (str): The ID of the experiment to fetch runs for.
            view_type (ViewType): The view type to filter runs (e.g., active,
                deleted, all).

        Returns:
            list[RunInfo]: A list of RunInfo objects representing the filtered runs.

        """
        stages = LifecycleStage.view_type_to_stages(view_type)
        if not stages:
            return []

        placeholders = ",".join(["?"] * len(stages))

        sql = f"""
            SELECT run_id, experiment_id, user_id, status, start_time, end_time,
                artifact_uri, lifecycle_stage
            FROM runs
            WHERE experiment_id = ? AND lifecycle_stage IN ({placeholders})
            ORDER BY start_time DESC
        """
        params = [experiment_id, *stages]
        rows = self.conn.execute(sql, params).fetchall()

        return [
            RunInfo(
                run_uuid=row[0],
                run_id=row[0],  # for backward compatibility
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
        """Search runs across multiple experiments using filters and sort criteria.

        This method retrieves runs from the provided list of experiment IDs,
        applies optional filtering, sorting, and pagination, and returns a
        paginated list of Run objects.

        Args:
            experiment_ids (list[str]): List of experiment IDs to search within.
            filter_string (str | None): A filter expression to apply to the results.
            run_view_type (ViewType): Specifies whether to include active, deleted,
                or all runs.
            max_results (int): Maximum number of runs to return.
            order_by (list[str] | None): Optional list of ordering criteria.
            page_token (str | None): Optional token for pagination.

        Returns:
            PagedList: A paginated list of Run objects matching the criteria.

        """
        runs: list[Run] = []

        # Retrieve and collect runs from each experiment
        for experiment_id in experiment_ids:
            run_infos = self._list_run_infos(experiment_id, run_view_type)
            for run_info in run_infos:
                runs.append(self.get_run(run_info.run_id))

        # Apply filtering, sorting, and pagination in memory
        filtered = SearchUtils.filter(runs, filter_string)
        sorted_runs = SearchUtils.sort(filtered, order_by)
        paginated, next_token = SearchUtils.paginate(runs=sorted_runs,
                                                     page_token=page_token,
                                                     max_results=max_results,
                                                    )

        return PagedList(paginated, token=next_token)

    def update_run_info(
        self,
        run_id: str,
        run_status: str,
        end_time: int,
        run_name: str,
    ) -> RunInfo:
        """Update the metadata of a run such as status, end time, and name.

        This method validates that the run is active, updates its status,
        end time, and optionally run name, and persists the changes.

        Args:
            run_id (str): The unique identifier of the run to update.
            run_status (str): The new status to set (e.g., 'FINISHED', 'FAILED').
            end_time (int): Epoch milliseconds for when the run ended.
            run_name (str): The new name to set for the run.

        Returns:
            RunInfo: The updated run metadata object.

        """
        run_info = self._get_run_info(run_id)

        # Ensure the run is in an active state before updating
        check_run_is_active(run_info)

        # Create a copy of the run info with updated values
        new_info = run_info._copy_with_overrides(
            status=run_status,
            end_time=end_time,
            run_name=run_name,
        )

        # Persist the updated run info to the database
        self._overwrite_run_info(new_info)

        return new_info

    def delete_run(self, run_id: str) -> None:
        """Mark a run as deleted by setting its lifecycle stage to 'deleted'.

        This method performs a soft delete, meaning the run is not removed from the
        database but marked as deleted by updating its lifecycle stage. Only active
        runs can be deleted.

        Args:
            run_id (str): The unique identifier of the run to delete.

        Raises:
            MlflowException: If the run is already deleted or not found.

        """
        run_info = self._get_run_info(run_id)

        # Only active runs can be deleted
        if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
            msg = f"Run '{run_id}' is already deleted."
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        # Set lifecycle_stage to DELETED and update last modified time
        new_info = run_info._copy_with_overrides(
            lifecycle_stage=LifecycleStage.DELETED,
        )

        self._overwrite_run_info(new_info, deleted_time=get_current_time_millis())

    def restore_run(self, run_id: str) -> None:
        """Restore a previously deleted run by setting its lifecycle stage to 'active'.

        This method performs a soft restore by updating the lifecycle stage from
        DELETED to ACTIVE. Only runs that are currently marked as deleted can
        be restored.

        Args:
            run_id (str): The unique identifier of the run to restore.

        Raises:
            MlflowException: If the run is not deleted or not found.

        """
        run_info = self._get_run_info(run_id)

        # Only deleted runs can be restored
        if run_info.lifecycle_stage != LifecycleStage.DELETED:
            msg = f"Run '{run_id}' is not deleted and cannot be restored."
            raise MlflowException(msg, RESOURCE_DOES_NOT_EXIST)

        # Set lifecycle_stage back to ACTIVE
        new_info = run_info._copy_with_overrides(
            lifecycle_stage=LifecycleStage.ACTIVE,
        )

        self._overwrite_run_info(new_info, deleted_time=get_current_time_millis())

    def _overwrite_run_info(self,
                            run_info: RunInfo,
                            deleted_time: int | None = None,
                            ) -> None:
        """Update the run metadata with the provided run info.

        This internal method updates the run's status, end time, lifecycle stage,
        and last update timein the database.

        Args:
            run_info (RunInfo): The updated run information.
            deleted_time (int | None): Optional override for the last update time.
                Defaults to current time.

        """
        last_update_time = deleted_time or get_current_time_millis()

        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, end_time = ?, lifecycle_stage = ?, last_update_time = ?
            WHERE run_id = ?
            """,
            (
                run_info.status,
                run_info.end_time,
                run_info.lifecycle_stage,
                last_update_time,
                run_info.run_id,
            ),
        )

    def log_inputs(self, run_id: str, datasets: list[DatasetInput]) -> None:
        """Log input datasets for a specific run.

        Args:
            run_id (str): The ID of the run to associate the datasets with.
            datasets (list[DatasetInput]): A list of dataset inputs to log, each
                containing a Dataset and optional tags.

        """
        for dataset_input in datasets:
            dataset = dataset_input.dataset
            tags_str = ",".join(f"{t.key}:{t.value}" for t in dataset_input.tags or [])

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
                    tags_str,
                ),
            )

    def _get_all_inputs(self, run_id: str) -> RunInputs:
        """Fetch all input datasets for a given run.

        Args:
            run_id (str): The ID of the run for which to retrieve inputs.

        Returns:
            RunInputs: A RunInputs object containing all DatasetInput entries
                associated with the run.

        """
        rows = self.conn.execute("""
            SELECT name, digest, source_type, source, schema, tags
            FROM datasets WHERE run_id = ?
        """, (run_id,)).fetchall()

        dataset_inputs = []
        for row in rows:
            name, digest, source_type, source, schema, tags_str = row

            # Parse tags from "key:value,key:value" string format
            tags = []
            if tags_str:
                for tag in tags_str.split(","):
                    if ":" in tag:
                        key, value = tag.split(":", 1)
                        tags.append(InputTag(key, value))

            dataset = Dataset(
                name=name,
                digest=digest,
                source_type=source_type,
                source=source,
                schema=schema,
            )

            dataset_inputs.append(DatasetInput(dataset=dataset, tags=tags))

        return RunInputs(dataset_inputs=dataset_inputs)

    def _search_datasets(self, experiment_ids: list[str]) -> list[_DatasetSummary]:
        """Return all unique dataset summaries associated with the given experiments.

        Datasets are deduplicated by (experiment_id, name, digest, context).
        Parsing is done in-memory using tags stored as "key:value" strings.

        Args:
            experiment_ids (list[str]): List of experiment IDs to scope the search.

        Returns:
            list[_DatasetSummary]: Unique dataset summaries used across the provided
                experiments.

        """
        seen: set[tuple[str, str, str, str | None]] = set()
        summaries: list[_DatasetSummary] = []

        placeholders = ",".join(["?"] * len(experiment_ids))
        query = f"""
            SELECT r.experiment_id, d.name, d.digest, d.tags
            FROM datasets d
            JOIN runs r ON d.run_id = r.run_id
            WHERE r.experiment_id IN ({placeholders})
        """
        rows = self.conn.execute(query, experiment_ids).fetchall()

        for experiment_id, name, digest, tags_str in rows:
            context: str | None = None

            # Extract the dataset context from the tags if available
            if tags_str:
                for tag in tags_str.split(","):
                    if ":" in tag:
                        key, value = tag.split(":", 1)
                        if key == MLFLOW_DATASET_CONTEXT:
                            context = value
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
