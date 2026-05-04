"""Simple test script to verify SimplifiedDiSCIPL works"""

import asyncio
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from true_working import SimplifiedPipeline
from evaluations.puzzle.tasks import SquareWordPoem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_simple():
    """Test with a simple task"""
    
    logger.info("="*80)
    logger.info("SimplifiedDiSCIPL - Simple Test")
    logger.info("="*80)
    
    # Create a simple task
    task = SquareWordPoem(N=4)  # 4x4 square poem
    logger.info(f"\nTask: {task.prompt}")
    
    # Initialize pipeline with smaller models for testing
    logger.info("\nInitializing pipeline...")
    
    pipeline = SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float16",
        max_tokens=64,
        max_attempts=2,
        results_dir="./test_results"
    )
    
    # Solve task
    logger.info("\nSolving task...")
    result = await pipeline.solve_task(task, use_examples=True)
    
    # Print result
    logger.info("\n" + "="*80)
    logger.info("RESULT")
    logger.info("="*80)
    logger.info(f"Task ID: {result.task_id}")
    logger.info(f"Is Valid: {result.is_valid}")
    logger.info(f"Attempts: {result.num_attempts}")
    logger.info(f"Time: {result.total_time:.2f}s")
    logger.info(f"\nGenerated Answer:\n{result.generated_answer}")
    logger.info(f"\nStrategy Used:")
    for i, step in enumerate(result.strategy, 1):
        logger.info(f"  {i}. {step}")


if __name__ == "__main__":
    asyncio.run(test_simple())
