"""Verify that SimplifiedDiSCIPL is ready to use"""

import sys
import os

print("SimplifiedDiSCIPL - Installation Verification")
print("=" * 80)

# Check Python version
print("\n1. Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ✗ Python {python_version.major}.{python_version.minor} (need 3.8+)")
    sys.exit(1)

# Check required packages
print("\n2. Checking required packages...")

packages = {
    "torch": "PyTorch",
    "transformers": "Hugging Face Transformers",
    "huggingface_hub": "Hugging Face Hub",
}

all_ok = True
for package_name, display_name in packages.items():
    try:
        __import__(package_name)
        print(f"   ✓ {display_name}")
    except ImportError:
        print(f"   ✗ {display_name} - NOT INSTALLED")
        all_ok = False

if not all_ok:
    print("\n   Install missing packages:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Check SimplifiedDiSCIPL structure
print("\n3. Checking SimplifiedDiSCIPL files...")

required_files = [
    "__init__.py",
    "planner.py",
    "follower.py",
    "pipeline.py",
    "quick_start.py",
    "run_benchmark.py",
]

current_dir = os.path.dirname(os.path.abspath(__file__))
all_present = True

for filename in required_files:
    filepath = os.path.join(current_dir, filename)
    if os.path.exists(filepath):
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename} - MISSING")
        all_present = False

if not all_present:
    print("\n   Some files are missing. Make sure you're in the true_working directory.")
    sys.exit(1)

# Try importing SimplifiedDiSCIPL
print("\n4. Testing imports...")

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
    from true_working import SimplifiedPipeline
    print("   ✓ SimplifiedPipeline imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import SimplifiedPipeline: {e}")
    sys.exit(1)

# Check GPU availability
print("\n5. Checking GPU...")

try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠ GPU not detected (will use CPU, but slower)")
except Exception as e:
    print(f"   ⚠ Could not check GPU: {e}")

# Verify puzzle tasks
print("\n6. Checking puzzle tasks...")

try:
    from evaluations.puzzle.tasks import load_all_tasks
    tasks = load_all_tasks()
    print(f"   ✓ Found {len(tasks)} puzzle tasks")
    for i, task in enumerate(tasks):
        print(f"     {i+1}. {task.prompt[:50]}...")
except Exception as e:
    print(f"   ✗ Failed to load puzzle tasks: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ ALL CHECKS PASSED!")
print("=" * 80)

print("\n📚 Next steps:")
print("  1. Read GETTING_STARTED.md for detailed instructions")
print("  2. Run: python quick_start.py")
print("  3. Or: python run_benchmark.py --num-tasks 2")

print("\n💾 Models will be downloaded on first use (~2-4GB)")
print("⏱️ First run may take 5-10 minutes for model download")

print("\nDocumentation:")
print("  - README.md          - Overview")
print("  - GETTING_STARTED.md - Detailed guide")
print("  - config_examples.py - Configuration templates")
