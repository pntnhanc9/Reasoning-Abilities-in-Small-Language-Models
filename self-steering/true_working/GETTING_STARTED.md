# GETTING STARTED - SimplifiedDiSCIPL

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
cd true_working
pip install torch transformers huggingface-hub
```

**Note:** This assumes you have a GPU with CUDA support. For CPU-only, use:
```bash
pip install torch transformers huggingface-hub --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Run Quick Test

```bash
python quick_start.py
```

This will:
1. Load the first puzzle task (4×4 square poem)
2. Download models if not cached (~2-4GB, ~5-10 minutes on first run)
3. Solve the task in ~20-60 seconds
4. Show the result

### Step 3: Check Results

Results are saved to `./test_results/`:

```bash
cat test_results/simplified_disciple_*/summary.json
```

---

## 📊 Run Full Benchmark

To test on multiple puzzle tasks:

```bash
python run_benchmark.py --num-tasks 4 --max-tokens 128
```

Parameters:
- `--num-tasks 4`: Run first 4 puzzle tasks
- `--max-tokens 128`: Allow up to 128 tokens per generation
- `--max-attempts 3`: Retry up to 3 times with feedback
- `--results-dir ./my_results`: Save to custom directory
- `--dtype float16`: Use float16 (default, saves memory)
- `--dtype float32`: Use float32 (better precision, more memory)

### Available Puzzle Tasks

From `evaluations/puzzle/tasks.py`:

1. **SquareWordPoem(N=8)**: 8×8 poem where each line has 8 words
   - Prompt: "Write a poem with N lines, where each line has exactly N words"
   - Evaluator: Checks exact line/word counts

2. **GrantProposal()**: Academic grant proposal with formatting
   - Prompt: Research proposal request
   - Evaluator: Checks structure/formatting

3. **IngredientsList()**: Recipe ingredients list
   - Prompt: List ingredients
   - Evaluator: Checks ingredient format

4. **TripItinerary()**: Travel itinerary planning
   - Prompt: Plan a trip
   - Evaluator: Checks day-by-day structure

---

## 🔧 Use in Your Own Code

### Basic Usage

```python
import asyncio
from true_working import SimplifiedPipeline
from evaluations.puzzle.tasks import SquareWordPoem

async def main():
    # Create pipeline
    pipeline = SimplifiedPipeline(
        max_tokens=128,
        max_attempts=3,
        results_dir="./my_results"
    )
    
    # Create task
    task = SquareWordPoem(N=4)
    
    # Solve
    result = await pipeline.solve_task(task)
    
    # Check result
    print(f"Valid: {result.is_valid}")
    print(f"Answer:\n{result.generated_answer}")

asyncio.run(main())
```

### Batch Processing

```python
import asyncio
from true_working import SimplifiedPipeline
from evaluations.puzzle.tasks import load_all_tasks

async def main():
    pipeline = SimplifiedPipeline()
    
    # Load all tasks
    tasks = load_all_tasks()[:5]  # First 5 tasks
    
    # Solve batch
    results = await pipeline.solve_batch(tasks)
    
    # Save summary
    pipeline.save_summary(results)

asyncio.run(main())
```

### Custom Configuration

```python
from config_examples import config_high_quality

# Use high-quality configuration
pipeline = config_high_quality()
result = await pipeline.solve_task(task)
```

---

## 💻 Available Configurations

See `config_examples.py` for detailed configurations:

### For Quick Testing
```python
from config_examples import config_minimal
pipeline = config_minimal()  # Uses ~2-4GB GPU
```

### For Default Use
```python
from config_examples import config_balanced
pipeline = config_balanced()  # Uses ~4-6GB GPU
```

### For Best Quality
```python
from config_examples import config_high_quality
pipeline = config_high_quality()  # Uses ~18GB GPU
```

### For Limited Memory
```python
from config_examples import config_low_memory
pipeline = config_low_memory()  # Uses ~4GB GPU
```

---

## 📁 File Structure

```
true_working/
├── __init__.py                 # Package init
├── planner.py                  # Strategy generator (replaces OpenAI)
├── follower.py                 # Text generator (replaces SMC)
├── pipeline.py                 # Main orchestration
├── run_benchmark.py            # Benchmark script
├── quick_start.py              # Quickest way to get started
├── test_simple.py              # Simple test
├── config_examples.py          # Configuration examples
├── README.md                   # Overview
└── GETTING_STARTED.md          # This file
```

---

## 🎯 Understanding the Pipeline

### 1. Planner Phase (~5-10s)

```
Input:  "Write a 4×4 square poem"
         + Examples (if available)

Planner:  "Generate step-by-step strategy"
          Uses local Llama model

Output: Strategy steps
        1. Understand: 4 lines, 4 words/line
        2. Plan: Use poetic language
        3. Generate: Write poem
        4. Validate: Check structure
```

### 2. Follower Phase (~10-30s)

```
Input:  Strategy + Task

Follower 1: Generate poem A
            (Llama 1B inference)

Follower 2: Generate poem B
            (Shared model, parallel)

Scoring:    A: score=1.2
            B: score=0.8

Output: Best result (A)
```

### 3. Validation Phase (~1s)

```
Input:  Generated poem

Validate: 
  ✓ Has 4 lines?
  ✓ Each line has 4 words?
  
Result:   VALID → Return
          INVALID → Retry with refined strategy
```

---

## ⚙️ Troubleshooting

### Problem: "CUDA out of memory"

**Solution 1**: Use smaller models
```python
pipeline = SimplifiedPipeline(
    planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",  # Already small
    follower_model="Qwen/Qwen3.5-0.8B",
)
```

**Solution 2**: Reduce max_tokens
```python
pipeline = SimplifiedPipeline(
    max_tokens=32,  # Reduce generation length
)
```

**Solution 3**: Use quantization (install bitsandbytes first)
```bash
pip install bitsandbytes
```

Then modify `follower.py` and `planner.py` to load models with quantization.

### Problem: "Very slow generation"

**Solution 1**: Reduce max_tokens
```python
pipeline = SimplifiedPipeline(max_tokens=32)
```

**Solution 2**: Disable some validation features

**Solution 3**: Use float32 instead of float16 (faster on some GPUs)
```python
pipeline = SimplifiedPipeline(dtype="float32")
```

### Problem: "Poor quality results"

**Solution 1**: Increase attempts (more retries)
```python
pipeline = SimplifiedPipeline(max_attempts=5)
```

**Solution 2**: Use larger planner
```python
pipeline = SimplifiedPipeline(
    planner_model="meta-llama/Llama-3.1-8B-Instruct"
)
```

**Solution 3**: Increase max_tokens
```python
pipeline = SimplifiedPipeline(max_tokens=256)
```

### Problem: "Models not downloading"

Check internet connection and try:
```bash
huggingface-cli login  # Login to HuggingFace
huggingface-cli download cyankiwi/Qwen3.5-9B-AWQ-4bit
huggingface-cli download Qwen/Qwen3.5-0.8B
```

---

## 📊 Comparing with Original DisCIPL

| Aspect | Original | SimplifiedDiSCIPL |
|--------|----------|------------------|
| **Planner** | GPT-4o API ($10) | Llama local ($0) |
| **Planner Speed** | 2-10s (API) | 5-10s (local) |
| **Followers** | 32 particles (SMC) | 2 followers (simple) |
| **GPU Memory** | 64GB | 18GB |
| **Cost** | $600-10K/1000 tasks | $0 |
| **Debuggability** | Hard (Python code) | Easy (text strategy) |
| **Setup Time** | Minutes | Hours (first model download) |

---

## 🎓 Learning the Code

### Key Classes

**LocalModelPlanner** (`planner.py`)
- Generates strategy instructions
- Methods:
  - `generate_strategy(task_prompt, examples)` - Generate new strategy
  - `refine_strategy(task_prompt, prev_strategy, error)` - Refine based on error

**SimplifiedFollower** (`follower.py`)
- Generates text following strategy
- Methods:
  - `generate(task_prompt, strategy)` - Generate one sample

**DualFollowerExecutor** (`follower.py`)
- Runs 2 followers in parallel
- Methods:
  - `generate_samples(task_prompt, strategy, num_samples)` - Generate N samples
  - `select_best(results, task_prompt)` - Select best by scoring
  - `score_result(result, task_prompt)` - Score a result

**SimplifiedPipeline** (`pipeline.py`)
- Main orchestration
- Methods:
  - `solve_task(task)` - Solve one task
  - `solve_batch(tasks)` - Solve multiple tasks
  - `save_summary(results)` - Save results to file

### Data Flow

```
Task
  ↓
SimplifiedPipeline.solve_task()
  ├→ LocalModelPlanner.generate_strategy()
  │   └→ Returns: {"strategy": [...], "reasoning": "..."}
  │
  ├→ DualFollowerExecutor.generate_samples()
  │   ├→ SimplifiedFollower.generate() × 2
  │   └→ Returns: [result1, result2]
  │
  ├→ DualFollowerExecutor.select_best()
  │   └→ Returns: best result
  │
  ├→ Task.evaluate(answer)
  │   └→ Returns: {evaluator_name: bool, ...}
  │
  └→ If invalid, retry loop ↑
  
Result: InferenceResult
  ├─ task_id
  ├─ generated_answer
  ├─ is_valid
  ├─ num_attempts
  ├─ total_time
  └─ ...
```

---

## 🚀 Next Steps

1. ✅ Run `quick_start.py` - Get familiar with basic usage
2. ✅ Run `run_benchmark.py` - Test on multiple tasks
3. ✅ Try different configurations from `config_examples.py`
4. ✅ Look at results in `./results/` folders
5. ✅ Modify code for your specific use cases
6. ✅ Integrate into your own projects

---

## 📚 Additional Resources

- **Original Paper**: https://arxiv.org/abs/2504.07081
- **LLaMPPL**: https://github.com/genlm/llamppl
- **Llama Models**: https://huggingface.co/meta-llama
- **HuggingFace Docs**: https://huggingface.co/docs
- **PyTorch Docs**: https://pytorch.org/docs

---

## ❓ FAQ

**Q: Do I need a GPU?**
A: For reasonable speed, yes. CPU-only will be 50-100x slower.

**Q: Which GPU do I need?**
A: Any GPU with 8GB+ VRAM. RTX 3060 or better recommended.

**Q: How long does first run take?**
A: Model download: 5-10 minutes (depends on internet)
   First inference: 30-60 seconds

**Q: Can I use larger models?**
A: Yes! Change `planner_model` to "meta-llama/Llama-3.1-8B-Instruct"
   Requires 16GB+ GPU.

**Q: Can I use smaller models?**
A: Yes! But quality may degrade. Llama 3.2 1B is already quite small.

**Q: How do I contribute improvements?**
A: See main project repository for contribution guidelines.

---

**Last Updated**: May 2026  
**Version**: 0.1.0  
**Status**: Ready for use
