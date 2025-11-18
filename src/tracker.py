import os
import json

class NoopTracker:
    def __init__(self, project=None, task_name=None, **kwargs):
        self.project = project
        self.task_name = task_name
        self.type = "none"

    def log_params(self, params: dict):
        return

    def log_metric(self, name: str, value, step: int = None):
        return

    def log_artifact(self, path: str, name: str = None):
        return

    def download_model(self, model_name: str = None, artifact_path: str = None):
        # If artifact_path is local path just return it
        if artifact_path and os.path.exists(artifact_path):
            return artifact_path
        return None

    def finalize(self):
        return


class ClearMLTracker:
    def __init__(self, project=None, task_name=None, **kwargs):
        try:
            from clearml import Task, OutputModel
        except Exception as e:
            raise ImportError("clearml not installed. Install with `pip install clearml` to use ClearML tracker.") from e
        self.Task = Task
        self.OutputModel = OutputModel
        self.project = project or "default"
        self.task_name = task_name or "task"
        # init task (auto-logs code, git etc when in git repo)
        self.task = Task.init(project_name=self.project, task_name=self.task_name)
        self.type = "clearml"

    def log_params(self, params: dict):
        self.task.connect(params)

    def log_metric(self, name: str, value, step: int = None):
        logger = self.task.get_logger()
        logger.report_scalar("metrics", name, value, iteration=step if step is not None else 0)

    def log_artifact(self, path: str, name: str = None):
        # name is optional; clearml will upload the file
        self.task.upload_artifact(name=name or os.path.basename(path), artifact_object=path)

    def download_model(self, model_name: str = None, artifact_path: str = None):
        """
        If model_name provided, fetch the model by name from ClearML OutputModel registry.
        If artifact_path is provided and local, return it.
        """
        if artifact_path and os.path.exists(artifact_path):
            return artifact_path
        if model_name:
            # Get latest model by name
            model = self.OutputModel.get_by_name(model_name, project_name=self.project)
            if model is None:
                raise RuntimeError(f"ClearML OutputModel '{model_name}' not found in project {self.project}")
            return model.get_local_copy()
        return None

    def finalize(self):
        # nothing special to do
        return


class MLflowTracker:
    def __init__(self, project=None, task_name=None, **kwargs):
        try:
            import mlflow
        except Exception as e:
            raise ImportError("mlflow not installed. Install with `pip install mlflow` to use MLflow tracker.") from e
        self.mlflow = mlflow
        self.project = project or "default"
        self.task_name = task_name or "task"
        # set experiment
        try:
            mlflow.set_experiment(self.project)
        except Exception:
            pass
        self.run = mlflow.start_run(run_name=self.task_name)
        self.type = "mlflow"

    def log_params(self, params: dict):
        try:
            self.mlflow.log_params(params)
        except Exception:
            # mlflow.log_params can fail if dict contains non-primitive - swallow safely
            for k, v in (params or {}).items():
                try:
                    self.mlflow.log_param(k, v)
                except Exception:
                    pass

    def log_metric(self, name: str, value, step: int = None):
        if step is None:
            self.mlflow.log_metric(name, float(value))
        else:
            self.mlflow.log_metric(name, float(value), step)

    def log_artifact(self, path: str, name: str = None):
        # mlflow stores artifacts under the run. It expects a file path.
        self.mlflow.log_artifact(path, artifact_path=name or "")

    def download_model(self, model_name: str = None, artifact_path: str = None):
        """
        For MLflow we support artifact_path (artifact_uri) that can be downloaded.
        If artifact_path is local path and exists return it. If artifact_path is an mlflow-artifact uri,
        try to download it via mlflow.artifacts.download_artifacts.
        """
        if artifact_path and os.path.exists(artifact_path):
            return artifact_path
        if artifact_path:
            try:
                local = self.mlflow.artifacts.download_artifacts(artifact_uri=artifact_path)
                return local
            except Exception as e:
                raise RuntimeError(f"Failed to download mlflow artifact {artifact_path}: {e}")
        # model registry integration would be done here in a real system
        return None

    def finalize(self):
        try:
            self.mlflow.end_run()
        except Exception:
            pass


def TrackerFactory(tracker_type: str = "none", project: str = None, task_name: str = None, **kwargs):
    tt = (tracker_type or "none").lower()
    if tt in ("none", "noop"):
        return NoopTracker(project=project, task_name=task_name, **kwargs)
    if tt in ("clearml", "clearmltracker"):
        return ClearMLTracker(project=project, task_name=task_name, **kwargs)
    if tt in ("mlflow", "mlflowtracker"):
        return MLflowTracker(project=project, task_name=task_name, **kwargs)
    raise ValueError(f"Unknown tracker type: {tracker_type}")
