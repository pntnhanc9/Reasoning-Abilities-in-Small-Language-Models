import os
import sys
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

import dill


class Task:
    def __init__(
        self,
        task_id: int,
        prompt: str,
        evaluators: Dict[str, Callable[[str], bool]],
        examples: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
    ):
        self._task_id = task_id
        self._prompt = prompt
        self._evaluators = evaluators
        self._examples = examples if examples is not None else []
        self._task_types = task_types if task_types is not None else []

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def evaluators(self) -> Dict[str, Callable[[str], bool]]:
        return self._evaluators

    @property
    def examples(self) -> List[str]:
        return self._examples

    @property
    def task_types(self) -> List[str]:
        return self._task_types

    def __repr__(self) -> str:
        line = "=" * 80
        return (
            f"{line}"
            f"\nTask(task_id={self.task_id}, task_types={self.task_types})"
            f"\n{self.prompt}\n"
            f"{line}"
        )

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_types": self.task_types,
            "prompt": self.prompt,
            "examples": self.examples,
        }

    def evaluate(self, text: str) -> Dict[str, bool]:
        return {name: evaluator(text) for name, evaluator in self.evaluators.items()}


class Dataset(ABC):
    def __init__(self, task_types: Optional[List[str]] = None):
        self.task_types = task_types
        self.tasks = self.load()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> List[Task]:
        pass

    def get_task_by_id(self, task_id: int) -> Task:
        matching_tasks = [task for task in self.tasks if task.task_id == task_id]
        if len(matching_tasks) == 0:
            raise ValueError(f"No task found with task_id={task_id}")
        elif len(matching_tasks) > 1:
            raise ValueError(f"Multiple tasks found with task_id={task_id}")
        return matching_tasks[0]

    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        return [task for task in self.tasks if task_type in task.task_types]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={len(self)}, task_types={self.task_types})"


class CollieDataset(Dataset):
    TASK_NAME_MAPPING = {
        # word-level tasks
        "english_c01": "word_01",
        "english_c02": "word_02",
        "english_c03": "word_03",
        # sentence-level tasks
        "wiki_c04": "sent_01",
        "wiki_c05": "sent_02",
        "wiki_c06a": "sent_03",
        "wiki_c07": "sent_04",
        # paragraph-level tasks
        "wiki_c08": "para_01",
        "wiki_c09": "para_02",
        "wiki_c10": "para_03",
        "wiki_c11": "para_04",
        "wiki_c12": "para_05",
    }

    def __init__(self, task_types: Optional[List[str]] = None):
        super().__init__(
            task_types=(
                task_types
                if task_types is not None
                else list(self.TASK_NAME_MAPPING.values())
            )
        )

    @property
    def name(self) -> str:
        return "collie"

    def load(self):
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "collie_eval",
                "Collie",
                "data",
                "all_data.dill",
            ),
            "rb",
        ) as f:
            collie_dataset = dill.load(f)

        tasks = []
        task_id = 0
        for task_name, task_type in self.TASK_NAME_MAPPING.items():
            for task in collie_dataset[task_name]:
                if task_type in self.task_types:
                    tasks.append(
                        Task(
                            task_id=task_id,
                            prompt=task["prompt"],
                            evaluators={
                                "valid": self.create_evaluator(
                                    task["constraint"], task["targets"]
                                )
                            },
                            examples=[task["example"]],
                            task_types=[task_type],
                        )
                    )
                task_id += 1

        return tasks

    def create_evaluator(self, constraint, targets):
        return lambda text: constraint(text, target=targets)


class IFEvalDataset(Dataset):
    def __init__(self, task_types: Optional[List[str]] = None):
        self.module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "ifeval", "google-research")
        )
        if not os.path.exists(self.module_path):
            raise FileNotFoundError(
                f"Could not find the instruction_following_eval module at {self.module_path}"
            )

        if self.module_path not in sys.path:
            sys.path.append(self.module_path)

        super().__init__(task_types=task_types)

    @property
    def name(self) -> str:
        return "ifeval"

    def load(self):
        from instruction_following_eval.evaluation_main import read_prompt_list

        task_list = read_prompt_list(
            os.path.join(
                self.module_path,
                "instruction_following_eval",
                "data",
                "input_data.jsonl",
            )
        )

        # If no task_types are specified, use all tasks
        if self.task_types is None:
            self.task_types = list(
                set(
                    instruction_id
                    for task in task_list
                    for instruction_id in task.instruction_id_list
                )
            )

        tasks = []

        for idx, task in enumerate(task_list):
            # Check if all task.instruction_id_list are in the task_types
            if all(
                task_type in self.task_types for task_type in task.instruction_id_list
            ):
                tasks.append(
                    Task(
                        task_id=idx,
                        prompt=task.prompt,
                        evaluators=self.create_evaluators(task),
                        task_types=task.instruction_id_list,
                    )
                )

        return tasks

    def create_evaluators(self, prompt):
        from instruction_following_eval.evaluation_main import (
            test_instruction_following_loose,
        )
        from instruction_following_eval.evaluation_main import (
            test_instruction_following_strict,
        )

        return {
            "strict": lambda text: test_instruction_following_strict(
                prompt, {prompt.prompt: text}
            ).follow_all_instructions,
            "loose": lambda text: test_instruction_following_loose(
                prompt, {prompt.prompt: text}
            ).follow_all_instructions,
        }


class PuzzleDataset(Dataset):

    def __init__(self, repeat: int = 1, **kwargs):
        self.repeat = repeat
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "puzzle"

    def load(self):
        from evaluations.puzzle.tasks import load_all_tasks

        original_tasks = load_all_tasks()

        tasks = []
        for task in original_tasks:
            for _ in range(self.repeat):
                t = deepcopy(task)
                t._task_id = len(tasks)
                tasks.append(t)
        return tasks


def load_dataset(
    name: str, task_types: Optional[List[str]] = None, repeat: int = 1
) -> Dataset:
    if name == "collie":
        return CollieDataset(task_types=task_types)
    elif name == "ifeval":
        return IFEvalDataset(task_types=task_types)
    elif name == "puzzle":
        return PuzzleDataset(task_types=task_types, repeat=repeat)
    else:
        raise ValueError(f"Unknown dataset: {name}")
