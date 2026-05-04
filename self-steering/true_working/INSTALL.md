# Installation Guide - SimplifiedDiSCIPL

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Disk Space**: ~10GB for models
- **Internet**: Required for first-time model download

## Installation Steps

### Step 1: Navigate to Directory

```bash
cd path/to/self-steering/true_working
```

### Step 2: Install Dependencies

#### Option A: Using pip (Recommended)

```bash
pip install -r requirements.txt
```

This installs:
- `torch >= 2.0.0` (PyTorch)
- `transformers >= 4.35.0` (Hugging Face)
- `huggingface-hub >= 0.19.0` (Model management)

#### Option B: Manual Installation

```bash
pip install torch transformers huggingface-hub
```

#### Option C: For CUDA (GPU Support)

If you have an NVIDIA GPU:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Option D: For CPU Only

```bash
pip install torch transformers huggingface-hub --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Verify Installation

```bash
python verify.py
```

This will check:
- Python version
- All required packages
- File structure
- GPU availability
- Puzzle tasks

You should see:
```
✓ ALL CHECKS PASSED!

Next steps:
  1. Read GETTING_STARTED.md
  2. Run: python quick_start.py
```

## Quick Test After Installation

### Test 1: Verify Setup
```bash
python verify.py
```

### Test 2: Run Quick Demo (30-60 seconds)
```bash
python quick_start.py
```

### Test 3: Run Full Benchmark
```bash
python run_benchmark.py --num-tasks 2
```

## Troubleshooting Installation

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch
```bash
pip install torch
```

### Issue: "ModuleNotFoundError: No module named 'transformers'"

**Solution**: Install Transformers
```bash
pip install transformers
```

### Issue: "CUDA out of memory" on first run

**Solution 1**: Use smaller model
```bash
# In quick_start.py or your script, change:
pipeline = SimplifiedPipeline(
    planner_model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
    follower_model="Qwen/Qwen3.5-0.8B",
    dtype="float16",  # Use float16 to save memory
    max_tokens=32,    # Reduce token generation
)
```

**Solution 2**: Install with quantization
```bash
pip install bitsandbytes
```

**Solution 3**: Check GPU memory
```bash
nvidia-smi
```

### Issue: "Model download failed"

**Solution**: Check internet connection
```bash
# Manual download
huggingface-cli download cyankiwi/Qwen3.5-9B-AWQ-4bit
huggingface-cli download Qwen/Qwen3.5-0.8B

# Or login to HuggingFace
huggingface-cli login
huggingface-cli download cyankiwi/Qwen3.5-9B-AWQ-4bit
huggingface-cli download Qwen/Qwen3.5-0.8B
```

### Issue: "Slow first run"

**Expected**: First run downloads models (~2-4GB, takes 5-10 minutes)

Subsequent runs will be fast.

### Issue: "Python version incompatible"

**Solution**: Upgrade Python
```bash
# On Windows
python --version  # Check current version

# Install Python 3.10+
# Download from python.org or use package manager
```

## GPU Configuration

### Check Your GPU

```bash
nvidia-smi
```

This shows:
- GPU model
- VRAM amount
- Current usage

### Recommended GPU VRAM

| Configuration | VRAM Needed | GPU Examples |
|---------------|------------|--------------|
| Minimal | 4GB | RTX 3050, RTX 4050 |
| Balanced | 8GB | RTX 3060, RTX 4060 |
| High Quality | 16GB+ | RTX 3080, RTX 4080 |

## Environment Variables (Optional)

Set these to customize behavior:

```bash
# Limit GPU
export CUDA_VISIBLE_DEVICES=0

# Set model cache directory
export HF_HOME=/path/to/models

# Set temp directory
export TMPDIR=/path/to/temp
```

## Docker Installation (Advanced)

If you prefer Docker:

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "quick_start.py"]
```

Build and run:
```bash
docker build -t simplified-disciple .
docker run --gpus all simplified-disciple
```

## Conda Installation (Alternative)

If you use Conda:

```bash
# Create environment
conda create -n simplified-disciple python=3.10

# Activate
conda activate simplified-disciple

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers huggingface-hub
```

## Virtual Environment Setup (Best Practice)

### Using venv

```bash
# Create virtual environment
python -m venv venv_simplified

# Activate (Linux/Mac)
source venv_simplified/bin/activate

# Activate (Windows)
venv_simplified\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify
python verify.py
```

### Using Poetry

```bash
# Install poetry
pip install poetry

# Install dependencies
poetry install

# Activate shell
poetry shell

# Verify
python verify.py
```

## Uninstallation

### Remove Installation

```bash
# If using virtual environment, deactivate first
deactivate

# Remove environment (optional)
rm -rf venv_simplified
```

### Clean Up Model Cache

```bash
# Default cache location
rm -rf ~/.cache/huggingface

# Or custom location
rm -rf $HF_HOME
```

## Verification Checklist

After installation, verify:

- [ ] Python version >= 3.8
- [ ] `pip install -r requirements.txt` completed
- [ ] `python verify.py` shows all ✓
- [ ] GPU detected (optional but recommended)
- [ ] Can import SimplifiedPipeline
- [ ] Puzzle tasks load successfully
- [ ] `python quick_start.py` runs without errors

## Next Steps

Once installation is verified:

1. **Read**: `GETTING_STARTED.md`
2. **Run**: `python quick_start.py`
3. **Explore**: `config_examples.py`
4. **Benchmark**: `python run_benchmark.py`

## Getting Help

If you encounter issues:

1. Check `verify.py` output
2. Read `GETTING_STARTED.md` troubleshooting section
3. Check system resources with `nvidia-smi`
4. Ensure internet connection for model download

## System-Specific Notes

### Windows
- May need to install "C++ Build Tools"
- Use `python` instead of `python3`
- Paths use backslash `\`

### macOS
- Works with Intel and Apple Silicon (slower on Apple Silicon)
- Use `python3` instead of `python`
- May need to install Xcode Command Line Tools

### Linux
- Most straightforward setup
- Ensure NVIDIA drivers installed for GPU
- Use `python3` and `pip3`

---

**Last Updated**: May 2026  
**Version**: 0.1.0
