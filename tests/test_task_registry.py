import pytest

from configs.tasks import ExtractionTask, available_tasks, load_task


def test_demo_task_registered():
    assert "demo_extraction" in available_tasks()


def test_load_demo_task():
    task = load_task("demo_extraction")
    assert isinstance(task, ExtractionTask)
    assert task.name == "demo_extraction"
    assert task.field_by_name("fever_present") is not None
    assert task.prompt_name == "demo_extraction"


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        load_task("does_not_exist")
