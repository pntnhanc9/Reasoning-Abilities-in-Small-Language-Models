class ModelRegistry:
    registry = {}

    @classmethod
    def register(
        cls,
        dataset: str,
        task_type: str,
        task_id: int,
        prompt: str = None,
        reasoning: str = None,
    ):
        def decorator(model_class):
            cls.registry[model_class.__name__] = {
                "dataset": dataset,
                "task_type": task_type,
                "task_id": task_id,
                "prompt": prompt,
                "reasoning": reasoning,
                "model_class": model_class,
            }
            return model_class

        return decorator

    def __getitem__(self, name: str):
        if name in self.registry:
            return self.registry[name]["model_class"]
        else:
            raise KeyError(f"Model '{name}' not found in the registry.")

    def get_models_by_task_type(self, task_type: str):
        return [
            entry["model_class"]
            for entry in self.registry.values()
            if entry["task_type"] == task_type
        ]

    def get_models_by_dataset(self, dataset: str):
        return [
            entry["model_class"]
            for entry in self.registry.values()
            if entry["dataset"] == dataset
        ]
