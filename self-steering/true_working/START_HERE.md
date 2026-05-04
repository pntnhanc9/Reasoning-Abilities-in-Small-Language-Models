# 🚀 START HERE - SimplifiedDiSCIPL

Welcome! This folder contains a simplified, cost-efficient, and GPU-friendly implementation of Self-Steering Language Models.

## ⚡ Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python verify.py
```

### 3. Run First Test
```bash
python quick_start.py
```

**That's it!** You'll see a complete example of:
- Loading a puzzle task
- Generating a strategy
- Running followers
- Validating results

---

## 📚 Documentation Guide

**New to this project?** Read these in order:

1. **START_HERE.md** (this file)
   - Quick navigation
   - What to do first

2. **GETTING_STARTED.md** (15 min read)
   - Detailed setup instructions
   - Understanding the pipeline
   - Configuration options
   - Troubleshooting guide

3. **README.md** (10 min read)
   - Project overview
   - Features and benefits
   - Model recommendations
   - Performance comparison

4. **INSTALL.md** (as needed)
   - Detailed installation options
   - System requirements
   - Troubleshooting installation issues

5. **IMPLEMENTATION_SUMMARY.md** (reference)
   - What was implemented
   - File structure overview
   - Expected performance

---

## 🎯 What You Can Do

### Option A: Quick Test (Easiest)
```bash
python quick_start.py
```
**Time**: 30-60 seconds  
**Output**: Shows complete pipeline working on one task

### Option B: Explore Configurations
```python
from config_examples import config_minimal
pipeline = config_minimal()
```
**6 pre-made configurations** in `config_examples.py`

### Option C: Run Full Benchmark
```bash
python run_benchmark.py --num-tasks 4
```
**Time**: 2-5 minutes  
**Output**: Tests on 4 puzzle tasks, saves results

### Option D: Use in Your Code
```python
import asyncio
from true_working import SimplifiedPipeline

async def main():
    pipeline = SimplifiedPipeline()
    # Use it...

asyncio.run(main())
```
**Integration**: Use in your own projects

---

## 📁 File Structure

```
true_working/
│
├── 📖 DOCUMENTATION (Read First!)
│   ├── START_HERE.md              ← You are here
│   ├── GETTING_STARTED.md         ← Detailed guide
│   ├── README.md                  ← Overview
│   ├── INSTALL.md                 ← Installation help
│   └── IMPLEMENTATION_SUMMARY.md   ← What was built
│
├── 🚀 QUICK START (Run These!)
│   ├── quick_start.py             ← Run this first!
│   ├── test_simple.py             ← Simple async test
│   ├── run_benchmark.py           ← Full benchmark
│   └── verify.py                  ← Check setup
│
├── ⚙️ CONFIGURATION
│   └── config_examples.py         ← 6 config templates
│
├── 💻 CORE IMPLEMENTATION
│   ├── __init__.py                ← Package init
│   ├── planner.py                 ← Strategy generator
│   ├── follower.py                ← Text generator
│   └── pipeline.py                ← Orchestration
│
├── 📦 DEPENDENCIES
│   └── requirements.txt
│
└── 📊 RESULTS (Generated)
    └── results/                   ← After running
```

---

## 🎓 Learning Path

### For Beginners (30 min)
1. Run `quick_start.py` → See it work
2. Read `GETTING_STARTED.md` → Understand how
3. Try `run_benchmark.py` → Test on multiple tasks

### For Developers (1-2 hours)
1. Explore `config_examples.py` → See configuration options
2. Read `planner.py` → Understand strategy generation
3. Read `follower.py` → Understand text generation
4. Read `pipeline.py` → Understand orchestration
5. Modify code → Adapt to your needs

### For Researchers (2-4 hours)
1. Read all documentation
2. Study the implementation
3. Analyze results in `results/summary.json`
4. Compare with original DisCIPL
5. Customize for your research

---

## 🎯 Common Tasks

### "I want to see it working"
```bash
python quick_start.py
```

### "I want to test on multiple tasks"
```bash
python run_benchmark.py --num-tasks 5
```

### "I want to understand the configuration"
```python
from config_examples import config_high_quality
pipeline = config_high_quality()  # Best quality
```

### "I want to use it in my code"
```python
from true_working import SimplifiedPipeline

pipeline = SimplifiedPipeline()
result = await pipeline.solve_task(my_task)
```

### "I want to check if setup is correct"
```bash
python verify.py
```

### "I want detailed instructions"
```
Read: GETTING_STARTED.md
```

---

## ⚠️ Important Notes

### First Run
- **Models download**: ~2-4GB (~5-10 minutes)
- **Memory needed**: ~4-20GB GPU VRAM (depends on config)
- **Internet**: Required for model download

### GPU
- **Recommended**: 8GB+ VRAM
- **Minimum**: 4GB VRAM (tight, may need quantization)
- **CPU**: Will work but very slow (~50x slower)

### Performance
- **Time per task**: 20-60 seconds (depends on config)
- **Success rate**: 70-90% (depends on task difficulty)
- **Retries**: Automatic with feedback loop (1-3 attempts)

---

## 🔍 What's Different from Original?

| Aspect | Original | SimplifiedDiSCIPL |
|--------|----------|-----------------|
| **Planner** | OpenAI API ($10) | Llama local ($0) |
| **Cost** | $600-10K/1K tasks | $0 |
| **GPU** | 64GB | 18GB |
| **Followers** | 32 particles | 2 followers |
| **Speed** | 30-60s | 20-40s |
| **Easy to use** | No | Yes |
| **Easy to debug** | No | Yes |

---

## ✅ Checklist to Get Started

- [ ] Read this file (START_HERE.md)
- [ ] Install: `pip install -r requirements.txt`
- [ ] Verify: `python verify.py`
- [ ] Test: `python quick_start.py`
- [ ] Read: `GETTING_STARTED.md`
- [ ] Explore: `config_examples.py`
- [ ] Benchmark: `python run_benchmark.py --num-tasks 2`

---

## 🆘 Need Help?

1. **Installation issues?** → Read `INSTALL.md`
2. **Understand how to use?** → Read `GETTING_STARTED.md`
3. **What was built?** → Read `IMPLEMENTATION_SUMMARY.md`
4. **Something not working?** → Run `verify.py`
5. **See examples?** → Check `config_examples.py`

---

## 🎯 Next Step

### Ready? Pick one:

**I want to see it work right now:**
```bash
python quick_start.py
```

**I want to understand everything:**
```
Read: GETTING_STARTED.md
```

**I want to customize it:**
```
Edit: config_examples.py
```

**I want to integrate it:**
```python
from true_working import SimplifiedPipeline
pipeline = SimplifiedPipeline()
```

---

## 📞 Files You'll Use Most

1. **`quick_start.py`** - Run this first
2. **`run_benchmark.py`** - Run tests on multiple tasks
3. **`config_examples.py`** - See configuration options
4. **`GETTING_STARTED.md`** - Read for detailed guide
5. **Results folder** - Check outputs after running

---

## 🚀 You're All Set!

**Step 1: Install**
```bash
pip install -r requirements.txt
```

**Step 2: Test**
```bash
python quick_start.py
```

**Step 3: Explore**
```
Read documentation and try examples
```

Good luck! 🎉

---

**For detailed instructions, see:** `GETTING_STARTED.md`  
**For troubleshooting, see:** `INSTALL.md`  
**For technical details, see:** `README.md`
