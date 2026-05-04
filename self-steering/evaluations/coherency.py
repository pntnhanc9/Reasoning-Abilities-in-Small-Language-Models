import asyncio
import json  # Use JSON for caching
import os
import re
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm


class CoherencyEvaluator:

    DEFAULT_BATCH_SIZE = 20
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

    def __init__(
        self,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = 0.0,
        cache_filename: str = None,
    ):
        load_dotenv()
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.openai_model = openai_model
        self.temperature = temperature
        self.init_cache(cache_filename)

    async def __call__(self, input_texts: List[str], batch_size: int = None):
        unrated_texts = [text for text in (set(input_texts)) if text not in self.cache]
        if unrated_texts:
            print(f"Rating {len(unrated_texts)} / {len(input_texts)} unrated texts...")
            await self.rate_batch(unrated_texts, batch_size)
        return await asyncio.gather(*[self.rate(text) for text in input_texts])

    def init_cache(self, cache_filename: str = None):
        """Set up the cache from DISCIPLE_CACHE or default directory."""
        cache_filename = cache_filename or "coherency.json"
        cache_dir = os.environ.get("DISCIPLE_CACHE", ".cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_file = os.path.join(cache_dir, cache_filename)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def clear_cache(self):
        self.cache = {}
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    @staticmethod
    def make_prompt(input_text: str):
        return f"Please concisely analyze the following text for coherency. Your analysis should be no longer than 3 sentences and it *must* end verbatim with 'The coherency score is <SCORE>', where <SCORE> is a number between 1 and 10.\n\n{input_text}"

    @staticmethod
    def extract_coherency_score(input_text: str):
        # Define a regular expression pattern to match the coherency score (case-insensitive)
        pattern = r"The coherency score is (\d+)"
        # Use re.search to find the match in the input string with case-insensitive flag
        match = re.search(pattern, input_text, flags=re.IGNORECASE)
        if match:
            # Extract and return the coherency score as a float
            coherency_score = int(match.group(1))
            return coherency_score
        else:
            # Return None if no match is found
            print(f"No match found in {input_text}")
            return None

    async def rate(self, input_text: str):
        # Check if the text is already in the cache
        if input_text in self.cache:
            return self.cache[input_text]

        if not input_text:
            return {
                "input_text": input_text,
                "analysis": "The input text is empty. The coherency score is 0.",
                "score": 0,
                "completion": None,
            }

        parameters = {
            "model": self.openai_model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": self.make_prompt(input_text),
                },
            ],
        }
        completion = await self.client.chat.completions.create(**parameters)
        message = completion.choices[0].message.content

        result = {
            "input_text": input_text,
            "analysis": message,
            "score": self.extract_coherency_score(message),
            "completion": completion.model_dump(),
        }
        self.cache[input_text] = result
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)
        return result

    async def rate_batch(self, input_texts: List[str], batch_size: int = None):
        """Rate a batch of input texts asynchronously."""
        if batch_size is None:
            batch_size = min(self.DEFAULT_BATCH_SIZE, len(input_texts))
        all_results = []
        total = len(input_texts)
        pbar = tqdm(total=total, desc="Evaluating coherency")
        for i in range(0, total, batch_size):
            batch = input_texts[i : i + batch_size]
            tasks = [self.rate(text) for text in batch]
            results = await asyncio.gather(*tasks)
            for result in results:
                all_results.append(result)
                pbar.update(1)
        pbar.close()
        return all_results

    @staticmethod
    def compute_cost(completion, prompt_token_price=0.15, completion_token_price=0.60):
        if completion is None:
            return 0
        return (
            (completion["usage"]["prompt_tokens"] * prompt_token_price)
            + (completion["usage"]["completion_tokens"] * completion_token_price)
        ) / 1e6
