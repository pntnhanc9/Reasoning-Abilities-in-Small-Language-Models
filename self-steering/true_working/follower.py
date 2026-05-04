"""Simplified Follower - Simple parallel generation instead of SMC"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class SimplifiedFollower:
    """Generate text following a strategy (simple, fast, efficient)
    
    This replaces the complex SMC inference with simple parallel generation.
    Instead of 32 particles with resampling, we use 2 simple followers.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        dtype: str = "float16",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        max_tokens: int = 32,
        temperature: float = 0.8,
    ):
        """Initialize SimplifiedFollower
        
        Args:
            model_name: HuggingFace model ID
            dtype: Data type (float16 or float32)
            device_map: Device placement strategy
            cache_dir: Cache directory for models
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        
        logger.info(f"Loading follower model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Load model
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"Follower model loaded: {model_name}")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate(
        self,
        task_prompt: str,
        strategy: List[str],
        sample_id: int = 0,
    ) -> Dict:
        """Generate a single sample following the strategy
        
        Args:
            task_prompt: The task description
            strategy: List of strategy steps from planner
            sample_id: Sample identifier
            
        Returns:
            Dict with generated text and metadata
        """
        
        # Build prompt using strategy
        prompt = self._build_generation_prompt(task_prompt, strategy)
        
        logger.debug(f"Sample {sample_id}: Generating with {len(strategy)} strategy steps")
        
        # Generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Extract answer
        answer = self._extract_answer(generated_text, task_prompt)
        
        return {
            "sample_id": sample_id,
            "full_text": generated_text,
            "answer": answer,
            "tokens": len(outputs.sequences[0]),
            "model": self.model_name,
        }
    
    def _build_generation_prompt(self, task_prompt: str, strategy: List[str]) -> str:
        """Build prompt that includes strategy steps"""
        
        strategy_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(strategy)])
        
        prompt = f"""Follow these steps to solve the task:

{strategy_text}

Task: {task_prompt}

Solution:"""
        
        return prompt
    
    def _extract_answer(self, text: str, task_prompt: str) -> str:
        """Extract just the answer portion from generated text"""
        
        # Look for common answer patterns
        if "<answer>" in text and "</answer>" in text:
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        if "Answer:" in text:
            parts = text.split("Answer:")
            if len(parts) > 1:
                return parts[-1].strip()
        
        if "Solution:" in text:
            parts = text.split("Solution:")
            if len(parts) > 1:
                return parts[-1].strip()
        
        # Otherwise return last non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return text.strip()


class DualFollowerExecutor:
    """Execute two followers in parallel and select best result"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        dtype: str = "float16",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        max_tokens: int = 32,
        temperature: float = 0.8,
    ):
        """Initialize with single model shared by both followers"""
        
        self.follower = SimplifiedFollower(
            model_name=model_name,
            dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    async def generate_samples(
        self,
        task_prompt: str,
        strategy: List[str],
        num_samples: int = 2,
    ) -> List[Dict]:
        """Generate multiple samples in sequence
        
        Args:
            task_prompt: The task description
            strategy: List of strategy steps
            num_samples: Number of samples to generate (default 2)
            
        Returns:
            List of generated samples
        """
        
        results = []
        for sample_id in range(num_samples):
            result = await self.follower.generate(
                task_prompt,
                strategy,
                sample_id=sample_id
            )
            results.append(result)
        
        return results
    
    def score_result(self, result: Dict, task_prompt: str) -> float:
        """Score a result based on heuristics
        
        Args:
            result: Generated sample
            task_prompt: Task description (for context)
            
        Returns:
            Score (0.0 to 2.0)
        """
        
        text = result.get("answer", "")
        
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer reasonable length)
        text_len = len(text.split())
        if 3 <= text_len <= 200:
            score += 1.0
        elif text_len >= 1:
            score += 0.5
        
        # No error keywords
        error_keywords = ["error", "unknown", "invalid", "fail", "none", "n/a"]
        has_errors = any(kw in text.lower() for kw in error_keywords)
        
        if not has_errors:
            score += 1.0
        else:
            score += 0.2
        
        # Bonus for longer texts (more detailed answers)
        if text_len > 10:
            score += 0.2
        
        return score
    
    def select_best(
        self,
        results: List[Dict],
        task_prompt: str,
    ) -> Dict:
        """Select best result from multiple samples
        
        Args:
            results: List of generated samples
            task_prompt: Task description
            
        Returns:
            Best result with score
        """
        
        scored_results = []
        for result in results:
            score = self.score_result(result, task_prompt)
            scored_results.append((result, score))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        best_result, best_score = scored_results[0]
        best_result["score"] = best_score
        best_result["rank"] = 0
        
        # Add rank to other results
        for i, (result, score) in enumerate(scored_results[1:], 1):
            result["score"] = score
            result["rank"] = i
        
        return best_result
