import asyncio
import string
from typing import List
from typing import Tuple

import llamppl
from llamppl import LMContext
from llamppl.distributions.lmcontext import LMNextToken
from llamppl.distributions.lmcontext import LMTokenMask

from disciple.masks import EOSMask
from disciple.masks import NewLineMask
from disciple.masks import NextTokenMask
from disciple.masks import PunctuationMask
from disciple.masks import TokenLengthMask


class BaseModel(llamppl.Model):
    """Base inference model."""

    def __init__(
        self,
        context: LMContext,
        max_tokens: int = 32,
    ):
        super().__init__()
        self.context = context
        self.max_tokens = max_tokens

        self.tokenizer = context.lm.tokenizer
        self.EOS_TOKEN_ID = context.lm.tokenizer.eos_token_id

    @classmethod
    async def create(
        cls,
        lm,
        task_prompt: str = None,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ):
        """Async factory method to create the model."""
        formatted_prompt = BaseModel.get_formatted_prompt(
            lm, user_prompt=cls.prior_prompt()
        )
        context = await LMContext.create(
            lm=lm,
            prompt=formatted_prompt,
            temp=temperature,
            show_eos=False,
        )
        model = cls(context, max_tokens)
        if task_prompt is not None:
            model.task_prompt = task_prompt
            formatted_task_prompt = BaseModel.get_formatted_prompt(
                lm, system_prompt=cls.system_prompt(), user_prompt=task_prompt
            )
            model.proposal_context = await LMContext.create(
                lm=lm, prompt=formatted_task_prompt, temp=temperature, show_eos=False
            )
        else:
            model.proposal_context = None
        return model

    @classmethod
    def system_prompt(cls):
        return (
            "You are helping a user generate text that satisfies constraints. "
            "Follow the user's instructions exactly. Write your response below; "
            "do not preface your response or include any additional remarks."
        )

    @classmethod
    def prior_prompt(cls):
        """A task-agnostic prompt that will be used to evaluate the prior probability of the generation."""
        raise NotImplementedError

    @staticmethod
    def get_formatted_prompt(
        lm,
        system_prompt: str = None,
        user_prompt: str = None,
        assistant_content: str = None,
    ):
        messages = BaseModel.get_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        formatted_prompt = lm.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if assistant_content is not None:
            formatted_prompt += assistant_content
        return formatted_prompt

    @staticmethod
    def get_chat_messages(
        system_prompt: str = None,
        user_prompt: str = None,
        system_role: str = "system",
    ):
        """Returns a list of messages in chat format."""
        messages = []
        if system_prompt is not None:
            messages += [
                {
                    "role": system_role,
                    "content": system_prompt,
                }
            ]
        if user_prompt is not None:
            messages += [{"role": "user", "content": user_prompt}]
        return messages

    def immutable_properties(self):
        return set(
            [
                "max_tokens",
                "tokenizer",
                "EOS_TOKEN_ID",
            ]
        )

    def __str__(self):
        return str(self.context)

    def __len__(self):
        return len(str(self.context))

    async def step(self):
        """Implements a single step of the generation process.

        NOTE: This method is task-specific and should be implemented by subclasses. See the examples for guidance.
        """
        raise NotImplementedError

    async def check(self, text: str) -> bool:
        """Implements a checking procedure that will be run at the end of generation.

        Args:
            text (str): The generated text.

        Returns:
            bool: True if the text is valid, False otherwise.

        NOTE: This method is task-specific and should be implemented by subclasses. See the examples for guidance.
        """
        raise NotImplementedError

    async def sample(self, dist: llamppl.Distribution):
        return await super().sample(dist, proposal=self.get_proposal_dist(dist))

    async def observe(self, dist: llamppl.Distribution, x):
        if self.proposal_context is not None:
            await self.intervene(self.get_proposal_dist(dist), x)
        return await super().observe(dist, x)

    async def hint(self, hint_text: str):
        if self.proposal_context is None:
            return
        hint_prompt = BaseModel.get_formatted_prompt(
            lm=self.proposal_context.lm,
            system_prompt=self.system_prompt(),
            user_prompt=self.task_prompt + f"\n\n(Note to self: {hint_text})",
            assistant_content=str(self.context),
        )
        hint_context = await LMContext.create(
            lm=self.proposal_context.lm,
            prompt=hint_prompt,
            temp=self.proposal_context.temp,
            show_eos=False,
        )
        await self.intervene(hint_context.mask_dist(self.context.model_mask), True)
        self.proposal_context = hint_context

    def get_proposal_dist(self, dist: llamppl.Distribution):
        if self.proposal_context is None:
            return None
        if isinstance(dist, LMNextToken):
            return self.proposal_context.next_token()
        elif isinstance(dist, LMTokenMask):
            return self.proposal_context.mask_dist(dist.mask)
        elif isinstance(dist, NextTokenMask):
            return self.proposal_context.mask_dist(dist.mask.mask)
        else:
            raise ValueError(f"Unsupported distribution type: {type(dist)}")

    async def next_token(self) -> str:
        """Generates a single token. This method automatically extends the context with the generated token.

        Returns:
            token: llamppl.Token object representing the sampled token
        """
        # NOTE: Yields control back to the event loop. Necessary to allow timeouts to work correctly when this method is called in a loop.
        await asyncio.sleep(0)
        return await self.sample(self.context.next_token())

    async def next_word(
        self,
        max_chars: int = None,
    ) -> str:
        """Generates a single word. This method automatically extends the context with the generated word.

        Args:
            max_chars (int): Maximum number of characters in the word. If None, the model will sample a word of any length.

        Returns:
            word: The sampled word.
        """
        await asyncio.sleep(0)

        # NOTE: This approach sometimes breaks with max_chars = 1
        if max_chars is not None:
            assert max_chars > 1

        last_token = (
            self.context.lm.str_vocab[self.context.tokens[-1]]
            if len(self.context.tokens) > 0
            else ""
        )
        last_character = last_token[-1] if len(last_token) > 0 else ""
        needs_space = (
            last_character not in string.whitespace
            and last_character
            not in [
                "-",
                "'",
                '"',
            ]
        )
        if needs_space:
            starts_word_mask = self.context.lm.masks.STARTS_NEW_WORD
        else:
            starts_word_mask = self.context.lm.masks.CONTINUES_CURRENT_WORD

        if len(self.context.model_mask.intersection(starts_word_mask)) > 0:
            # Force model to start a new word
            await self.observe(self.context.mask_dist(starts_word_mask), True)

        word = ""
        while True:
            # Force model to sample a token with an appropriate number of characters
            if max_chars is not None:
                await self.observe(
                    self.context.mask_dist(
                        self.context.lm.masks.token_length_mask(
                            max=max_chars - len(word.strip())
                        )
                    ),
                    True,
                )

            token = await self.next_token()
            word += self.context.lm.str_vocab[token.token_id]

            # If we ran out of chars, break
            if max_chars is not None and len(word.strip()) >= max_chars:
                await self.observe(
                    self.context.mask_dist(
                        self.context.lm.masks.CONTINUES_CURRENT_WORD
                    ),
                    False,
                )
                break

            # If the model wants to end the word, break
            if not (
                await self.sample(
                    self.context.mask_dist(self.context.lm.masks.CONTINUES_CURRENT_WORD)
                )
            ):
                break

        # Optionally, sample mid-word punctuation (commas, colons, hyphens, quotes, etc.)
        if await self.sample(
            self.context.mask_dist(self.context.lm.masks.MID_PUNCTUATION)
        ):
            token = await self.next_token()
            word += self.context.lm.str_vocab[token.token_id]

        return word

    def _add_prefix_space(self, text: str) -> str:
        """Adds a prefix space to the text if it does not already start with one.
        Does not add a space if the context is empty or already ends with whitespace.

        Args:
            text (str): Text to add a prefix space to.

        Returns:
            text: Text with a prefix space added.
        """
        context = str(self.context)
        # Add a prefix space
        if (
            not text.startswith(" ")
            and not context.endswith(tuple(string.whitespace))
            and len(context) > 0
        ):
            text = " " + text
        return text

    async def extend_with(self, text: str, add_prefix_space: bool = True) -> str:
        """Extends the generation with a pre-defined string literal. This method automatically extends the context with the input text.

        Args:
            text (str): String to extend the generation with.
            add_prefix_space (bool): Auto-add a prefix space to text. In most cases, this should be left as True.

        Returns:
            text: The generated text (same as input).
        """
        await asyncio.sleep(0)

        if add_prefix_space:
            text = self._add_prefix_space(text)

        for token_id in self.tokenizer.encode(text, add_special_tokens=False):
            await self.observe(self.context.next_token(), token_id)

        return text

    async def extend(
        self,
        start: str = None,
        stop: List[str] = None,
        min_chars: int = None,
        max_chars: int = None,
        allow_eos: bool = True,
        add_prefix_space: bool = True,
    ) -> Tuple[str, bool]:
        """Extends the generation with a new string. This method automatically extends the context with the generated text.

        Args:
            start (str): String to start the generation with.
            stop (List[str]): List of strings to stop the generation at.
            min_chars (int): Minimum number of characters to generate.
            max_chars (int): Maximum number of characters to generate.
            allow_eos (bool): Allow EOS token to be generated.
            add_prefix_space (bool): Auto-add a prefix space to text. In most cases, this should be left as True.

        Returns:
            new_text: The generated text.
            eos: Whether the generation was stopped by an EOS token.
        """
        await asyncio.sleep(0)

        assert isinstance(start, (str, type(None)))
        assert isinstance(stop, (list, str, type(None)))
        assert isinstance(max_chars, (int, type(None)))
        assert isinstance(min_chars, (int, type(None)))

        if max_chars and min_chars:
            assert max_chars >= min_chars

        if isinstance(stop, str):
            stop = [stop]

        old_text = str(self.context)

        if start is not None:
            start = await self.extend_with(start, add_prefix_space=add_prefix_space)

        new_text = start or ""
        eos = False

        for _ in range(self.context.token_count, self.max_tokens):
            async with TokenLengthMask(
                self,
                max_chars=max_chars - len(new_text) if max_chars is not None else None,
                allow_eos=allow_eos and len(new_text) >= (min_chars or 0),
            ):
                token = await self.next_token()
            new_text = str(self.context)[len(old_text) :]

            # Stop on EOS token.
            if int(token) == self.EOS_TOKEN_ID:
                eos = True
                break

            # Stop on character limit.
            if max_chars is not None and len(new_text) >= max_chars:
                break

            # Stop on any stop string.
            if (
                stop is not None
                and any([s in new_text for s in stop])
                and len(new_text) >= (min_chars or 0)
            ):
                break

        return new_text, eos

    async def end(self):
        """Mark the generation as finished. This method automatically appends an EOS token if it has not already been generated."""
        if self.context.tokens[-1] != self.EOS_TOKEN_ID:
            async with EOSMask(self):
                await self.next_token()
        self.finish()
