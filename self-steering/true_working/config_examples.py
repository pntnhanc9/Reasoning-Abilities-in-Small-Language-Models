"""
Configuration examples for SimplifiedDiSCIPL

This file shows different configurations for different use cases.
"""

from true_working import SimplifiedPipeline


# ============================================================================
# CONFIGURATION 1: Minimal (For Testing - Fastest, Uses Least Memory)
# ============================================================================

def config_minimal():
    """Minimal configuration - fast and low memory"""
    return SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float16",
        max_tokens=32,
        max_attempts=1,
        results_dir="./results_minimal"
    )


# ============================================================================
# CONFIGURATION 2: Balanced (Default - Good Quality and Speed)
# ============================================================================

def config_balanced():
    """Balanced configuration - good quality and reasonable speed"""
    return SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float16",
        max_tokens=64,
        max_attempts=2,
        results_dir="./results_balanced"
    )


# ============================================================================
# CONFIGURATION 3: High Quality (Larger Planner for Better Reasoning)
# ============================================================================

def config_high_quality():
    """High quality configuration - better reasoning with larger planner"""
    return SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",  # 9B instead of 1B
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float16",
        max_tokens=128,
        max_attempts=3,
        results_dir="./results_high_quality"
    )


# ============================================================================
# CONFIGURATION 4: Low Memory (For GPU with Limited VRAM)
# ============================================================================

def config_low_memory():
    """Low memory configuration - for GPUs with limited VRAM (~8GB)"""
    return SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float16",
        max_tokens=16,  # Reduce generation length
        max_attempts=1,  # Single attempt
        results_dir="./results_low_memory"
    )


# ============================================================================
# CONFIGURATION 5: Float32 (For Better Precision on Some Architectures)
# ============================================================================

def config_float32():
    """Float32 configuration - better precision but more memory"""
    return SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        dtype="float32",  # Full precision instead of float16
        max_tokens=64,
        max_attempts=2,
        results_dir="./results_float32"
    )


# ============================================================================
# CONFIGURATION 6: Research (Best Quality - No Constraints)
# ============================================================================

def config_research():
    """Research configuration - best possible quality"""
    return SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",  # Large planner
        follower_model="Qwen/Qwen3.5-0.8B",  # Or larger if available
        dtype="float16",
        max_tokens=256,  # Allow longer generation
        max_attempts=5,  # More retries for perfect answer
        results_dir="./results_research"
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    print("SimplifiedDiSCIPL Configuration Examples")
    print("="*80)
    
    configs = {
        "Minimal": {
            "use_case": "Quick testing, limited resources",
            "gpu_memory": "~2-4GB",
            "speed": "Very Fast (5-10s per task)",
            "quality": "Basic",
            "fn": config_minimal
        },
        "Balanced": {
            "use_case": "Default, good for most use cases",
            "gpu_memory": "~4-6GB",
            "speed": "Fast (10-20s per task)",
            "quality": "Good",
            "fn": config_balanced
        },
        "High Quality": {
            "use_case": "When quality matters most",
            "gpu_memory": "~18-20GB",
            "speed": "Medium (20-40s per task)",
            "quality": "Excellent",
            "fn": config_high_quality
        },
        "Low Memory": {
            "use_case": "Limited GPU VRAM (~8GB)",
            "gpu_memory": "~4-6GB",
            "speed": "Very Fast",
            "quality": "Basic",
            "fn": config_low_memory
        },
        "Float32": {
            "use_case": "Better precision on some GPUs",
            "gpu_memory": "~8-12GB",
            "speed": "Medium",
            "quality": "Good",
            "fn": config_float32
        },
        "Research": {
            "use_case": "Best possible results",
            "gpu_memory": "~20GB",
            "speed": "Slow (40-60s per task)",
            "quality": "Best",
            "fn": config_research
        },
    }
    
    for name, info in configs.items():
        print(f"\n{name}:")
        print(f"  Use Case:   {info['use_case']}")
        print(f"  GPU Memory: {info['gpu_memory']}")
        print(f"  Speed:      {info['speed']}")
        print(f"  Quality:    {info['quality']}")
    
    print("\n" + "="*80)
    print("\nHow to use:")
    print("  1. Choose a configuration above")
    print("  2. Import the function: from config_examples import config_balanced")
    print("  3. Create pipeline: pipeline = config_balanced()")
    print("  4. Use as normal: result = await pipeline.solve_task(task)")
    print("\nExample:")
    print("""
import asyncio
from config_examples import config_balanced
from evaluations.puzzle.tasks import SquareWordPoem

async def main():
    pipeline = config_balanced()
    task = SquareWordPoem(N=4)
    result = await pipeline.solve_task(task)
    print(f"Valid: {result.is_valid}")
    print(f"Answer: {result.generated_answer}")

asyncio.run(main())
""")
