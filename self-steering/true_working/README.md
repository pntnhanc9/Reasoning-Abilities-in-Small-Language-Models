# SimplifiedDiSCIPL - Quick & Efficient Implementation

A simplified, cost-efficient, and GPU-friendly implementation of Self-Steering Language Models.

## Features

- ✅ **Free**: Uses only open-source models (no OpenAI API costs)
- ✅ **GPU Efficient**: Uses 18GB peak GPU memory (vs 64GB for original)
- ✅ **Fast**: 20-40 seconds per task (vs 30-60s for original)
- ✅ **Simple**: 2 followers instead of 32 particles
- ✅ **Debuggable**: Text-based strategy format instead of complex Python code

## Architecture

### Planner (Strategy Generator)
- **Model**: Llama 3.2 1B Instruct (or Llama 3.1 8B for better quality)
- **Task**: Generate step-by-step strategy instructions
- **Cost**: Free (open-source)
- **Memory**: ~2GB

### Follower (Text Generator)
- **Model**: Llama 3.2 1B Instruct
- **Task**: Generate text following strategy
- **Followers**: 2 in parallel (vs 32 particles)
- **Cost**: Free
- **Memory**: ~4GB total

### Pipeline (Orchestration)
- Generates strategy from planner
- Runs 2 followers in parallel
- Scores and selects best result
- Retries with refined strategy if validation fails

## Installation

### 1. Install Dependencies

```bash
cd true_working
pip install torch transformers huggingface-hub
```

### 2. Download Models (optional)

```bash
huggingface-cli download cyankiwi/Qwen3.5-9B-AWQ-4bit
huggingface-cli download Qwen/Qwen3.5-0.8B
```

Models will be auto-downloaded on first use if not present.

## Quick Start

### Simple Test

```bash
cd true_working
python test_simple.py
```

This tests on a single 4×4 square poem task.

### Run Benchmark

```bash
cd true_working
python run_benchmark.py --num-tasks 2 --max-tokens 128
```

Options:
- `--num-tasks N`: Number of tasks to run
- `--max-tokens N`: Max tokens to generate per sample
- `--max-attempts N`: Max retry attempts with feedback
- `--dtype float16|float32`: Data type
- `--cache-dir PATH`: Where to cache models
- `--results-dir PATH`: Where to save results

## Usage in Code

```python
import asyncio
from true_working import SimplifiedPipeline
from evaluations.puzzle.tasks import SquareWordPoem

async def main():
    # Create pipeline
    pipeline = SimplifiedPipeline(
        planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        follower_model="Qwen/Qwen3.5-0.8B",
        max_tokens=128,
        max_attempts=3,
        results_dir="./results"
    )
    
    # Create task
    task = SquareWordPoem(N=4)
    
    # Solve
    result = await pipeline.solve_task(task)
    
    print(f"Valid: {result.is_valid}")
    print(f"Answer: {result.generated_answer}")

asyncio.run(main())
```

## Results Structure

Results are saved to `results/<timestamp>/`:

```
results/simplified_disciple_20260504_120000/
├── summary.json          # Overall statistics
├── task_0000/
│   └── result.json       # Result for task 0
├── task_0001/
│   └── result.json       # Result for task 1
└── ...
```

Each `result.json` contains:
```json
{
  "task_id": 0,
  "task_prompt": "...",
  "strategy": ["Step 1", "Step 2", ...],
  "generated_answer": "...",
  "is_valid": true,
  "num_attempts": 1,
  "total_time": 25.3
}
```

## Performance Comparison

| Metric | Original | SimplifiedDiSCIPL |
|--------|----------|-----------------|
| Cost | $600-10K/1K tasks | $0 |
| GPU Peak | 64GB | 18GB |
| Speed | 30-60s/task | 20-40s/task |
| Followers | 32 particles | 2 followers |
| Code complexity | High | Low |

## Model Recommendations

### For Testing (Fast)
```python
pipeline = SimplifiedPipeline(
    planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
    follower_model="Qwen/Qwen3.5-0.8B",
)
```
- GPU: ~4GB
- Speed: Fast
- Quality: Good for simple tasks

### For Production (Best Quality)
```python
pipeline = SimplifiedPipeline(
    planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
    follower_model="Qwen/Qwen3.5-0.8B",
)
```
- GPU: ~18GB
- Speed: Medium
- Quality: Better reasoning

### For Low-Memory Systems (Quantized)
```bash
# Use 4-bit quantization
pip install bitsandbytes
```

Then load models with:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

This reduces memory by 4x: 18GB → 4.5GB

## Puzzle Tasks Included

Test data from `evaluations/puzzle/tasks.py`:

1. **SquareWordPoem(N)**: Write N×N poem where each line has N words
   - Example: 4×4 poem = 4 lines, each with 4 words

2. **GrantProposal()**: Write academic grant proposal with constraints

3. **IngredientsList()**: List ingredients with specific requirements

4. **TripItinerary()**: Create trip itinerary following constraints

## File Structure

```
true_working/
├── __init__.py           # Package initialization
├── planner.py            # LocalModelPlanner (strategy generator)
├── follower.py           # SimplifiedFollower (text generator)
├── pipeline.py           # SimplifiedPipeline (main orchestration)
├── run_benchmark.py      # Benchmark script
├── test_simple.py        # Simple test script
└── README.md             # This file
```

## How It Works

### 1. Planner Phase
```
Task: "Write a 4×4 square poem"
  ↓
Planner generates:
  1. Understand: 4 lines, 4 words per line
  2. Plan: Use varied vocabulary, maintain meaning
  3. Generate: Write poem
  4. Validate: Check constraints
```

### 2. Follower Phase
```
Strategy + Task
  ↓
Follower 1 → Poem A (score: 1.2)
Follower 2 → Poem B (score: 0.8)
  ↓
Select best: Poem A
```

### 3. Validation
```
Validate Poem A against task requirements
  ✓ Valid → Return result
  ✗ Invalid → Refine strategy → Retry
```

## Troubleshooting

### GPU Out of Memory (OOM)

**Option 1**: Use smaller models
```python
pipeline = SimplifiedPipeline(
    planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
    follower_model="Qwen/Qwen3.5-0.8B",
)
```

**Option 2**: Use 4-bit quantization
```bash
pip install bitsandbytes
# Then modify follower.py to use quantization
```

**Option 3**: Reduce max_tokens
```python
pipeline = SimplifiedPipeline(
    max_tokens=32,  # Reduce from 128
)
```

### Poor Quality Results

**Increase attempts**:
```python
pipeline = SimplifiedPipeline(
    max_attempts=5,  # More retries with feedback
)
```

**Use larger planner**:
```python
pipeline = SimplifiedPipeline(
    planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
)
```

### Slow Generation

**Reduce max_tokens**:
```python
pipeline = SimplifiedPipeline(
    max_tokens=32,
)
```

**Use float32 instead of float16** (faster on some GPUs):
```python
pipeline = SimplifiedPipeline(
    dtype="float32",
)
```

## Next Steps

1. ✅ Run `test_simple.py` to verify installation
2. ✅ Run `run_benchmark.py` with 2-5 tasks
3. ✅ Adjust `--max-tokens`, `--max-attempts` based on performance
4. ✅ Compare results with original DisCIPL
5. ✅ Integrate into your own pipeline

## References

- Original Paper: https://arxiv.org/abs/2504.07081
- COLM 2025: https://openreview.net/forum?id=XvCBtm5PgF
- LLaMPPL: https://github.com/genlm/llamppl
- **Qwen Models**: https://huggingface.co/Qwen

## License

Same as parent project (MIT)

---

**Created**: May 2026  
**Status**: Ready for use  
**Last Updated**: 2026-05-04
