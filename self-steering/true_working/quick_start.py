"""
Quick start guide for SimplifiedDiSCIPL

This file shows the simplest way to get started.
"""

import asyncio
import os
import sys

# Make sure we can import from parent directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from true_working import SimplifiedPipeline
from evaluations.puzzle.tasks import load_all_tasks


async def main():
    """Simplest example to get started"""
    
    print("\n" + "="*80)
    print("SimplifiedDiSCIPL - Quick Start")
    print("="*80 + "\n")
    
    # Step 1: Load a task
    print("Step 1: Loading puzzle task...")
    all_tasks = load_all_tasks()
    task = all_tasks[0]  # SquareWordPoem
    print(f"  Task: {task.prompt}")
    print(f"  Example: {task.examples[0] if task.examples else 'N/A'}\n")
    
    # Step 2: Create pipeline
    print("Step 2: Creating SimplifiedPipeline...")
    print("  This will download models if not cached (~2-4GB)")
    print("  Planner: Qwen 3.5 9B AWQ 4bit")
    print("  Follower: Qwen 3.5 0.8B")
    
    pipeline = SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float16",
        max_tokens=64,
        max_attempts=2,
        results_dir="./test_results"
    )
    print("  ✓ Pipeline ready\n")
    
    # Step 3: Solve task
    print("Step 3: Solving task (this may take 20-60 seconds)...")
    result = await pipeline.solve_task(task)
    print("  ✓ Task solved\n")
    
    # Step 4: Show results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Valid:    {result.is_valid}")
    print(f"Attempts: {result.num_attempts}")
    print(f"Time:     {result.total_time:.2f} seconds")
    print(f"\nGenerated Answer:")
    print("-" * 80)
    print(result.generated_answer)
    print("-" * 80)
    
    if result.strategy:
        print(f"\nStrategy Used:")
        for i, step in enumerate(result.strategy, 1):
            print(f"  {i}. {step}")
    
    print("\n" + "="*80)
    print(f"Result: {'✓ SUCCESS' if result.is_valid else '✗ FAILED'}")
    print("="*80 + "\n")
    
    # Step 5: Try another task
    print("Want to try another task?")
    print(f"  All available tasks: {len(all_tasks)}")
    for i, t in enumerate(all_tasks):
        print(f"    {i}. {t.prompt[:60]}...")
    print("\nModify the script and change task = all_tasks[0] to another index.\n")


if __name__ == "__main__":
    print("\nNote: First run will download models (~2-4GB)")
    print("      This may take a few minutes depending on your internet speed.\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Check GPU memory: nvidia-smi")
        print("  2. Install required packages: pip install torch transformers")
        print("  3. Check internet connection (for model download)")
        print("  4. See README.md for more help")
