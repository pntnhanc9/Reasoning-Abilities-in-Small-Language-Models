"""Run benchmark tests on SimplifiedDiSCIPL"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from true_working import SimplifiedPipeline
from evaluations.puzzle.tasks import load_all_tasks


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main benchmark function"""
    
    parser = argparse.ArgumentParser(
        description="Run SimplifiedDiSCIPL benchmark on puzzle tasks"
    )
    
    parser.add_argument(
        "--planner-model",
        type=str,
        default="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        help="Planner model (default: Qwen 3.5 9B AWQ 4bit)"
    )
    
    parser.add_argument(
        "--follower-model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Follower model"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Data type for models"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retry attempts"
    )
    
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=2,
        help="Number of tasks to run (default: 2 for quick test)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: results/timestamp)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join("results", f"simplified_disciple_{timestamp}")
    
    logger.info(f"Results will be saved to: {args.results_dir}")
    
    # Initialize pipeline
    logger.info("Initializing SimplifiedPipeline...")
    
    pipeline = SimplifiedPipeline(
        planner_model=args.planner_model,
        follower_model=args.follower_model,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
        results_dir=args.results_dir,
    )
    
    # Load puzzle tasks
    logger.info("Loading puzzle tasks...")
    all_tasks = load_all_tasks()
    tasks = all_tasks[:args.num_tasks]
    
    logger.info(f"Loaded {len(tasks)} tasks for testing")
    for i, task in enumerate(tasks):
        logger.info(f"  Task {i+1}: {task.prompt[:60]}...")
    
    # Run benchmark
    logger.info(f"\nStarting benchmark with {len(tasks)} tasks...")
    results = await pipeline.solve_batch(tasks, use_examples=True)
    
    # Save summary
    pipeline.save_summary(results)
    
    # Print results summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)
    
    valid_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)
    success_rate = valid_count / total_count if total_count > 0 else 0
    avg_time = sum(r.total_time for r in results) / total_count if total_count > 0 else 0
    
    logger.info(f"Total tasks: {total_count}")
    logger.info(f"Valid tasks: {valid_count}/{total_count} ({success_rate*100:.1f}%)")
    logger.info(f"Average time: {avg_time:.2f}s per task")
    
    logger.info("\nDetailed Results:")
    logger.info("-" * 80)
    
    for result in results:
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        logger.info(f"{status} | Task {result.task_id} | Attempts: {result.num_attempts} | Time: {result.total_time:.2f}s")
        logger.info(f"  Answer: {result.generated_answer[:80]}{'...' if len(result.generated_answer) > 80 else ''}")
        logger.info("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
