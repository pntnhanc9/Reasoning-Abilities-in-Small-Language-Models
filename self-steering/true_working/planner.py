"""Local Model Planner - Generates strategy instructions instead of LLaMPPL code"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalModelPlanner:
    """Generates Strategy Instructions via local Llama model
    
    This replaces the expensive OpenAI GPT-4o Planner.
    Instead of generating complex LLaMPPL Python code, we generate
    simple step-by-step strategy instructions in plain text.
    """
    
    def __init__(
        self,
        model_name: str = "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        dtype: str = "float16",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize LocalModelPlanner
        
        Args:
            model_name: HuggingFace model ID (default: Llama 3.2 1B, smaller for testing)
            dtype: Data type (float16 or float32)
            device_map: Device placement strategy
            cache_dir: Cache directory for models
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        
        logger.info(f"Loading model: {model_name}")
        
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
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        logger.info(f"Model loaded successfully: {model_name}")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_strategy(
        self,
        task_prompt: str,
        examples: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate a strategy for solving a task
        
        Args:
            task_prompt: The task description
            examples: List of example task-solution pairs (optional)
            
        Returns:
            Dict with "strategy" (list of steps) and "reasoning"
        """
        
        # Build prompt
        prompt = self._build_strategy_prompt(task_prompt, examples)
        
        logger.debug(f"Planner input length: {len(prompt.split())}")
        
        # Generate strategy
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract strategy from response
        strategy = self._parse_strategy(generated_text)
        
        logger.debug(f"Generated strategy: {strategy['strategy']}")
        
        return strategy
    
    def _build_strategy_prompt(
        self,
        task_prompt: str,
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """Build prompt for strategy generation
        
        The prompt is much simpler than the original LLaMPPL prompt.
        """
        
        examples_text = self._format_examples(examples) if examples else ""
        
        prompt = f"""You are an expert planner. Given a constrained generation task, 
generate a clear step-by-step strategy to solve it.

Your strategy should have 3-5 steps that clearly describe how to approach the problem.
Be specific about constraints and how to handle them.

{examples_text}

TASK: {task_prompt}

Strategy (provide 3-5 numbered steps):"""
        
        return prompt
    
    def _format_examples(self, examples: Optional[List[Dict]]) -> str:
        """Format examples for the prompt"""
        
        if not examples:
            return ""
        
        examples_text = "EXAMPLES:\n"
        for i, example in enumerate(examples[:2], 1):  # Use at most 2 examples
            task = example.get("prompt", "")
            solution = example.get("solution", "")
            
            examples_text += f"\nExample {i}: {task}\n"
            if solution:
                examples_text += f"Solution: {solution}\n"
        
        examples_text += "\n---\n\n"
        return examples_text
    
    def _parse_strategy(self, response_text: str) -> Dict:
        """Parse strategy from LLM response
        
        Extract numbered steps from the response.
        """
        
        steps = []
        
        # Extract numbered steps (1. ..., 2. ..., etc.)
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for numbered items
            if re.match(r'^\d+\.', line):
                # Remove numbering and leading whitespace
                step = re.sub(r'^\d+\.\s*', '', line).strip()
                if step and len(step) > 3:  # Only add non-trivial steps
                    steps.append(step)
        
        # If no steps found, use a default strategy
        if not steps:
            steps = [
                "Understand the task requirements and constraints",
                "Plan an approach that satisfies all constraints",
                "Generate candidate solution",
                "Validate solution against all constraints",
                "Output final answer"
            ]
        
        return {
            "strategy": steps,
            "reasoning": response_text
        }
    
    def refine_strategy(
        self,
        task_prompt: str,
        previous_strategy: List[str],
        error_message: str,
    ) -> Dict:
        """Refine strategy based on failed attempt
        
        Args:
            task_prompt: The task description
            previous_strategy: The previous strategy that failed
            error_message: Description of what went wrong
            
        Returns:
            Refined strategy
        """
        
        strategy_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(previous_strategy)])
        
        prompt = f"""You are an expert planner. The following strategy failed to solve a task:

TASK: {task_prompt}

PREVIOUS STRATEGY:
{strategy_text}

WHAT WENT WRONG: {error_message}

Provide a REVISED strategy (3-5 numbered steps) that avoids the previous failure:

Revised Strategy:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refined_strategy = self._parse_strategy(generated_text)
        
        logger.debug(f"Refined strategy: {refined_strategy['strategy']}")
        
        return refined_strategy
