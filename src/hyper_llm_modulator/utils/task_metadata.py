import os

import yaml

CULTURES_DIRECTORY = "cultures"
TASKS_DIRECTORY = "tasks"


def get_metadata(ds_names, use_per_task_emb, is_culture=False):
    metadata = dict()
    for ds_name in ds_names:
        metadata[ds_name] = get_metadata_for_task(ds_name, is_culture)
        if use_per_task_emb:
            assert "descriptions" in metadata[ds_name], "descriptions must be provided for either none or all datasets"
    return metadata


def get_metadata_for_task(task_name: str, is_culture=False) -> dict:
    """Return metadata for a single task."""
    metadata = {}
    task_dir = os.path.join(CULTURES_DIRECTORY if is_culture else TASKS_DIRECTORY, task_name)
    with open(os.path.join(task_dir, "metadata.yaml")) as f:
        metadata = yaml.safe_load(f.read())
        # Add task name based on the directory name.
        metadata["task_name"] = os.path.basename(task_dir)
    return metadata


def get_all_metadata(is_culture) -> list:
    """Return metadata for all tasks, sorted alphabetically."""
    # Get all task directories.
    task_dirs = []
    DIR_NAME = CULTURES_DIRECTORY if is_culture else TASKS_DIRECTORY
    for dir in os.listdir(DIR_NAME):
        if os.path.isdir(os.path.join(DIR_NAME, dir)):
            task_dirs.append(os.path.join(DIR_NAME, dir))

    metadata_list = []
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        metadata_list.append(get_metadata_for_task(task_name, is_culture))

    return sorted(metadata_list, key=lambda x: x["task_name"])


def get_all_metadata_as_dict(is_culture=False) -> dict:
    """Return metadata for all tasks as a dictionary."""
    metadata = get_all_metadata(is_culture)
    metadata_dict = {}
    for task in metadata:
        metadata_dict[task["task_name"]] = task
    return metadata_dict
