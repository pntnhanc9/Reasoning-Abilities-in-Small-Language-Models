"""Simplified Pipeline - Main orchestration with feedback loop"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from evaluations.dataset import Task

from .follower import DualFollowerExecutor
from .planner import LocalModelPlanner

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from a single inference"""
    
    task_id: int
    task_prompt: str
    strategy: List[str]
    generated_answer: str
    full_text: str
    score: float
    is_valid: bool
    num_attempts: int
    total_time: float
    model_name: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class SimplifiedPipeline:
    """Main pipeline combining Planner, Follower, and Feedback Loop"""
    
    def __init__(
        self,
        planner_model: str = "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model: str = "Qwen/Qwen3.5-0.8B",
        dtype: str = "float16",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        max_tokens: int = 32,
        max_attempts: int = 3,
        results_dir: Optional[str] = None,
    ):
        """Initialize SimplifiedPipeline
        
        Args:
            planner_model: Model for generating strategies
            follower_model: Model for generating answers
            dtype: Data type (float16 or float32)
            device_map: Device placement
            cache_dir: Cache directory
            max_tokens: Max tokens to generate
            max_attempts: Maximum retry attempts
            results_dir: Directory to save results
        """
        
        self.planner = LocalModelPlanner(
            model_name=planner_model,
            dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        
        self.follower = DualFollowerExecutor(
            model_name=follower_model,
            dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            max_tokens=max_tokens,
        )
        
        self.max_attempts = max_attempts
        self.results_dir = results_dir or "./results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"SimplifiedPipeline initialized")
        logger.info(f"  Planner: {planner_model}")
        logger.info(f"  Follower: {follower_model}")
        logger.info(f"  Max attempts: {max_attempts}")
    
    async def solve_task(
        self,
        task: Task,
        use_examples: bool = True,
    ) -> InferenceResult:
        """Solve a task with feedback loop
        
        Args:
            task: The task to solve
            use_examples: Whether to use examples from task
            
        Returns:
            InferenceResult
        """
        
        start_time = time.time()
        task_id = task.task_id if task.task_id is not None else 0
        
        logger.info(f"Solving task {task_id}: {task.prompt[:50]}...")
        
        # Prepare examples
        examples = None
        if use_examples and task.examples:
            examples = [{"prompt": task.prompt, "solution": ex} for ex in task.examples]
        
        strategy_state = {
            "task": task.prompt,
            "attempt": 0,
            "strategy": None,
            "previous_failures": []
        }
        
        best_result = None
        best_score = -1
        
        for attempt in range(self.max_attempts):
            strategy_state["attempt"] = attempt + 1
            
            logger.info(f"  Attempt {attempt + 1}/{self.max_attempts}")
            
            try:
                # STEP 1: Generate strategy
                if attempt == 0:
                    # First attempt: generate fresh strategy
                    strategy_result = self.planner.generate_strategy(
                        task.prompt,
                        examples=examples
                    )
                else:
                    # Retry: refine strategy based on feedback
                    failure_info = strategy_state["previous_failures"][-1]
                    strategy_result = self.planner.refine_strategy(
                        task.prompt,
                        strategy_state["strategy"],
                        failure_info.get("error", "Validation failed")
                    )
                
                strategy = strategy_result["strategy"]
                strategy_state["strategy"] = strategy
                
                logger.debug(f"  Strategy: {strategy}")
                
                # STEP 2: Generate samples with followers
                results = await self.follower.generate_samples(
                    task.prompt,
                    strategy,
                    num_samples=2
                )
                
                # STEP 3: Select best result
                best_candidate = self.follower.select_best(
                    results,
                    task.prompt
                )
                
                logger.debug(f"  Best answer: {best_candidate['answer'][:50]}...")
                
                # STEP 4: Validate result
                eval_results = task.evaluate(best_candidate["answer"])
                is_valid = all(eval_results.values())
                
                logger.info(f"  Validation: {eval_results}")
                
                if is_valid:
                    best_result = best_candidate
                    best_score = best_candidate["score"]
                    
                    logger.info(f"✓ Task {task_id} solved successfully in attempt {attempt + 1}")
                    break
                else:
                    # Record failure for feedback
                    error_msg = f"Validation failed: {eval_results}"
                    strategy_state["previous_failures"].append({
                        "attempt": attempt + 1,
                        "generated": best_candidate["answer"],
                        "error": error_msg,
                        "evaluation": eval_results
                    })
                    
                    logger.info(f"  Retrying with refined strategy...")
            
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {e}")
                strategy_state["previous_failures"].append({
                    "attempt": attempt + 1,
                    "error": str(e)
                })
        
        elapsed_time = time.time() - start_time
        
        # Prepare result
        if best_result is not None:
            is_valid = all(task.evaluate(best_result["answer"]).values())
            
            result = InferenceResult(
                task_id=task_id,
                task_prompt=task.prompt,
                strategy=strategy_state["strategy"],
                generated_answer=best_result["answer"],
                full_text=best_result.get("full_text", ""),
                score=best_result["score"],
                is_valid=is_valid,
                num_attempts=strategy_state["attempt"],
                total_time=elapsed_time,
                model_name=self.follower.follower.model_name,
            )
        else:
            # Failed all attempts
            result = InferenceResult(
                task_id=task_id,
                task_prompt=task.prompt,
                strategy=strategy_state["strategy"] or [],
                generated_answer="",
                full_text="",
                score=0.0,
                is_valid=False,
                num_attempts=self.max_attempts,
                total_time=elapsed_time,
                model_name=self.follower.follower.model_name,
            )
            
            logger.warning(f"✗ Task {task_id} failed after {self.max_attempts} attempts")
        
        return result
    
    async def solve_batch(
        self,
        tasks: List[Task],
        use_examples: bool = True,
    ) -> List[InferenceResult]:
        """Solve a batch of tasks sequentially
        
        Args:
            tasks: List of tasks
            use_examples: Whether to use examples
            
        Returns:
            List of results
        """
        
        results = []
        
        for i, task in enumerate(tasks):
            logger.info(f"\nProcessing task {i+1}/{len(tasks)}")
            
            result = await self.solve_task(task, use_examples=use_examples)
            results.append(result)
            
            # Save intermediate results
            self._save_result(result)
        
        return results
    
    def _save_result(self, result: InferenceResult) -> None:
        """Save result to file"""
        
        task_dir = os.path.join(self.results_dir, f"task_{result.task_id:04d}")
        os.makedirs(task_dir, exist_ok=True)
        
        result_path = os.path.join(task_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_summary(
        self,
        results: List[InferenceResult],
        summary_path: Optional[str] = None
    ) -> None:
        """Save summary of results"""
        
        if summary_path is None:
            summary_path = os.path.join(self.results_dir, "summary.json")
        
        # Calculate statistics
        total_tasks = len(results)
        valid_tasks = sum(1 for r in results if r.is_valid)
        success_rate = valid_tasks / total_tasks if total_tasks > 0 else 0
        avg_time = sum(r.total_time for r in results) / total_tasks if total_tasks > 0 else 0
        avg_attempts = sum(r.num_attempts for r in results) / total_tasks if total_tasks > 0 else 0
        
        summary = {
            "total_tasks": total_tasks,
            "valid_tasks": valid_tasks,
            "success_rate": success_rate,
            "avg_time_seconds": avg_time,
            "avg_attempts": avg_attempts,
            "results": [r.to_dict() for r in results]
        }
        
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nSummary saved to {summary_path}")
        logger.info(f"  Total tasks: {total_tasks}")
        logger.info(f"  Valid tasks: {valid_tasks}/{total_tasks} ({success_rate*100:.1f}%)")
        logger.info(f"  Avg time: {avg_time:.2f}s")
        logger.info(f"  Avg attempts: {avg_attempts:.2f}")
