"""
models.py
Author: grandg@mit.edu

Collection of hand-authored inference models for the Collie benchmark.

NOTE: These are written to accept constraints and targets as inputs.
"""

import hfppl as hp
from hfppl.chunks import sample_word
from hfppl.chunks import sample_word_2


def get_model_for_task(task_name):
    if task_name.endswith("c04"):
        return CollieModelSent01
    elif task_name.endswith("c05"):
        return CollieModelSent02
    elif task_name.endswith("c06a"):
        return CollieModelSent03
    elif task_name.endswith("c07"):
        return CollieModelSent04
    elif task_name.endswith("c08"):
        return CollieModelPara01
    elif task_name.endswith("c09"):
        return CollieModelPara02
    elif task_name.endswith("c10"):
        return CollieModelPara03
    elif task_name.endswith("c11"):
        return CollieModelPara04
    elif task_name.endswith("c12"):
        return CollieModelPara05
    else:
        raise ValueError(f"Task {task_name} not recognized.")


class CollieModelBase(hp.Model):
    """Base class for Collie models."""

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.context = hp.LMContext(llm, prompt, temperature, show_eos=False)
        self.tokenizer = llm.tokenizer
        self.vocab = llm.vocab

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.EOS_TOKEN = llm.tokenizer.eos_token_id

        self.constraints = constraints
        self.targets = targets

    def immutable_properties(self):
        return set(
            [
                "tokenizer",
                "vocab",
                "max_tokens",
                "temperature",
                "EOS_TOKEN",
                "constraints",
                "targets",
            ]
        )

    def __str__(self):
        return str(self.context)


class CollieModelRejectionSampling(CollieModelBase):
    """Rejection sampling model for the Collie benchmark."""

    async def step(self):
        """Each step samples an entire completion."""

        while True:
            # Sample token from language model
            token = await self.sample(self.context.next_token())

            # On EOS, finish
            if token == self.EOS_TOKEN:
                # Check the constraint
                self.condition(self.constraints(str(self.context), target=self.targets))
                self.finish()
                return

            # If we've run out of tokens, fail
            if self.context.token_count > self.max_tokens:
                self.condition(False)
                return


class CollieModelSent01(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a sentence with exactly C characters. Include whitespace into your character count.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.max_length = targets

    def immutable_properties(self):
        return super().immutable_properties().union(set(["max_length"]))

    async def step(self):
        """Each step samples a token from the language model."""

        current_length = len(str(self.context))
        remaining_length = self.max_length - current_length

        # Sample token from masked distribution of tokens that fit within remaining length
        await self.observe(
            self.context.mask_dist(
                self.context.lm.masks.MAX_TOKEN_LENGTH[remaining_length]
            ),
            True,
        )
        token = await self.sample(self.context.next_token())

        # On EOS, enforce that we have reached the character limit
        if remaining_length == 0:
            self.finish()
            return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelSent02(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a sentence:
    1) with exactly W words;
    2) with the i'th, j'th, k'th words to be 'word_1', 'word_2', 'word_3' respectively.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.word_idx = 0

        # Positions of constrained words; e.g., [3, 7, 10]
        self.target_idxs = constraints.callables[1].transformation.position

        # e.g., [16, ['by', 'gathering', 'as']]
        self.word_count, self.target_words = targets

        self.target_idx_to_word = dict(zip(self.target_idxs, self.target_words))

    def immutable_properties(self):
        return (
            super()
            .immutable_properties()
            .union(
                set(["target_idxs", "target_idx_to_word", "word_count", "target_words"])
            )
        )

    async def step(self):
        """Each step samples a word from the language model."""

        if self.word_idx in self.target_idxs:
            # Observe a constrained word
            for token in self.tokenizer.encode(
                " " + self.target_idx_to_word[self.word_idx], add_special_tokens=False
            ):
                await self.observe(self.context.next_token(), token)

            # TODO: Observe that the word is completed

        else:
            # Sample a word unconstrained
            word, mid_punctuation, end_punctuation = await self.call(
                sample_word_2(context=self.context, allow_end_punctuation=False)
            )

        self.word_idx += 1

        # If we've run out of words, force the model to finish the sentence
        if self.word_idx >= self.word_count:
            # Sample punctuation
            await self.observe(
                self.context.mask_dist(self.context.lm.masks.END_PUNCTUATION), True
            )
            await self.sample(self.context.next_token())
            # Observe EOS
            await self.observe(self.context.next_token(), self.EOS_TOKEN)
            self.finish()
            return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelSent03(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a sentence:
    1) with at least W words;
    2) with all words having at most C characters.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.min_words, self.max_chars_per_word = targets

        self.word_idx = 1

    def immutable_properties(self):
        return (
            super()
            .immutable_properties()
            .union(set(["min_words", "max_chars_per_word"]))
        )

    async def step(self):
        """Each step samples a word from the language model."""

        # Sample a word
        word, mid_punctuation, end_punctuation = await self.call(
            sample_word_2(
                context=self.context,
                max_chars=self.max_chars_per_word,
                allow_end_punctuation=(self.word_idx >= self.min_words),
            )
        )
        self.word_idx += 1

        # If we have enough words, allow the model to end the sentence
        if self.word_idx >= self.min_words:
            # If we encounter end-of-sentence punctuation, finish
            if end_punctuation:
                await self.observe(self.context.next_token(), self.EOS_TOKEN)
                self.finish()
                return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelSent04(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a sentence containing the words 'word_1', 'word_2', 'word_3'.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.target_words_remaining = [word.lower() for word in targets]

    def immutable_properties(self):
        return super().immutable_properties().union(set())

    async def step(self):
        """Each step samples multiple words until a constrained word is sampled."""

        # Sample a word unconstrained
        # If no target words remain, allow end punctuation
        word, mid_punctuation, end_punctuation = await self.call(
            sample_word_2(
                context=self.context,
                allow_mid_punctuation=True,
                allow_end_punctuation=not self.target_words_remaining,
            )
        )

        # If the word is a target word, remove it from the list
        word_standardized = word.strip().lower()
        if word_standardized in self.target_words_remaining:
            self.target_words_remaining = [
                w for w in self.target_words_remaining if w != word_standardized
            ]

        # If we encounter end-of-sentence punctuation, perform final check and force EOS
        if end_punctuation:
            # Ensure no target words remaining
            self.condition(not self.target_words_remaining)
            # Force the model to generate EOS
            await self.observe(self.context.next_token(), self.EOS_TOKEN)
            self.finish()
            return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelPara01(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a paragraph with all sentences having the 1st word to be 'word_1'.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.target_word = targets

    def immutable_properties(self):
        return super().immutable_properties().union(set(["target_word"]))

    async def step(self):
        """Each step samples a sentence from the language model."""

        # First, observe the target word
        for token in self.tokenizer.encode(
            " " + self.target_word, add_special_tokens=False
        ):
            await self.observe(self.context.next_token(), token)

        # TODO: Observe that the word is completed?

        # Sample the rest of the sentence word-by-word
        while True:
            # Sample a word unconstrained
            word, mid_punctuation, end_punctuation = await self.call(
                sample_word_2(context=self.context, allow_end_punctuation=True)
            )

            # If we encounter end-of-sentence punctuation, allow the model to optionally end the paragraph
            if end_punctuation:
                if await self.sample(self.context.mask_dist(set([self.EOS_TOKEN]))):
                    await self.sample(self.context.next_token())
                    self.finish()
                return

            # If we've run out of tokens, fail
            if self.context.token_count > self.max_tokens:
                self.condition(False)
                return


class CollieModelPara02(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a paragraph:
    1) with exactly S sentences;
    2) not containing the word 'word_1';
    3) not containing the word 'word_2';
    4) not containing the word 'word_3'.

    NOTE: The paper states that constraint #1 is "with at least S sentences".
    However, in the dataset, the constraint is "with exactly S sentences", which is what we implement here.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.n_sentences_target = targets[0]
        self.disallowed_words = set([word.lower() for word in targets[1:]])

        self.sentence_count = 0

    def immutable_properties(self):
        return (
            super()
            .immutable_properties()
            .union(set(["n_sentences_target", "disallowed_words"]))
        )

    async def step(self):
        """Each step samples a sentence from the language model."""

        # Sample the sentence word-by-word.
        while True:
            # Sample a word. Only allow end punctuation if we have enough words in the sentence.
            word, mid_punctuation, end_punctuation = await self.call(
                sample_word_2(
                    context=self.context,
                    allow_end_punctuation=True,
                )
            )

            # If the word is disallowed, fail
            if word.lower().strip() in self.disallowed_words:
                self.condition(False)
                return

            if end_punctuation:
                break

        self.sentence_count += 1

        # If we reach the sentence target, force the model to end the paragraph.
        if self.sentence_count >= self.n_sentences_target:
            await self.observe(self.context.mask_dist(set([self.EOS_TOKEN])), True)
            await self.sample(self.context.next_token())
            self.finish()
            return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelPara03(CollieModelBase):
    """Inference model for the following task.

        -- Task description --
    Please generate a paragraph:
    1) with exactly S sentences;
    2) with all sentences having at least W words;
    3) with all sentences having at most M words.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        (
            self.n_sentences_target,
            self.min_words_per_sentence,
            self.max_words_per_sentence,
        ) = targets

        self.sentence_count = 0

    def immutable_properties(self):
        return (
            super()
            .immutable_properties()
            .union(set(["min_sentences", "min_words_per_sentence"]))
        )

    async def step(self):
        """Each step samples a sentence from the language model."""

        # Sample the sentence word-by-word.
        word_count = 0
        while True:
            # If we've reached the maximum word count, reject.
            if word_count >= self.max_words_per_sentence:
                self.condition(False)
                return

            # Sample a word. Only allow end punctuation if we have enough words in the sentence.
            word, mid_punctuation, end_punctuation = await self.call(
                sample_word_2(
                    context=self.context,
                    allow_end_punctuation=(word_count >= self.min_words_per_sentence),
                )
            )
            word_count += 1

            if end_punctuation:
                break

        self.sentence_count += 1

        # If we reach the sentence target, force the model to end the paragraph.
        if self.sentence_count >= self.n_sentences_target:
            await self.observe(self.context.mask_dist(set([self.EOS_TOKEN])), True)
            await self.sample(self.context.next_token())
            self.finish()
            return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelPara04(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a paragraph:
    1) with at least S sentences;
    2) with all sentences having at least W words.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.min_sentences, self.min_words_per_sentence = targets

        self.sentence_count = 0

    def immutable_properties(self):
        return (
            super()
            .immutable_properties()
            .union(set(["min_sentences", "min_words_per_sentence"]))
        )

    async def step(self):
        """Each step samples a sentence from the language model."""

        # Sample the sentence word-by-word.
        word_count = 0
        while True:
            # Sample a word. Only allow end punctuation if we have enough words in the sentence.
            word, mid_punctuation, end_punctuation = await self.call(
                sample_word_2(
                    context=self.context,
                    allow_end_punctuation=(word_count >= self.min_words_per_sentence),
                )
            )
            word_count += 1

            if end_punctuation:
                break

        self.sentence_count += 1

        # If we have enough sentences, allow the model to optionally end the paragraph.
        if self.sentence_count >= self.min_sentences:
            if await self.sample(self.context.mask_dist(set([self.EOS_TOKEN]))):
                await self.sample(self.context.next_token())
                self.finish()
            return

        # If we've run out of tokens, fail
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            return


class CollieModelPara05(CollieModelBase):
    """Inference model for the following task.

    -- Task description --
    Please generate a paragraph:
    1) with exactly S sentences;
    2) with sentences having the last word to be 'word_1', 'word_2', ..., respectively.

    """

    def __init__(
        self,
        llm,
        prompt,
        constraints,
        targets,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            prompt=prompt,
            constraints=constraints,
            targets=targets,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Task-specific variables
        self.n_sentences_target = targets[0]
        self.final_word_per_sentence = [word.lower() for word in targets[1]]

        self.sentence_count = 0

    def immutable_properties(self):
        return (
            super()
            .immutable_properties()
            .union(set(["n_sentences_target", "final_word_per_sentence"]))
        )

    async def step(self):
        """Each step samples a sentence from the language model."""

        # Sample the sentence word-by-word.
        while True:
            # Sample a word. Disallow end punctuation.
            word, mid_punctuation, end_punctuation = await self.call(
                sample_word_2(
                    context=self.context,
                    allow_end_punctuation=False,
                )
            )

            # If we encounter the final word of the sentence, break.
            if (
                word.strip().lower()
                == self.final_word_per_sentence[self.sentence_count]
            ):
                break

            # If we've run out of tokens, fail.
            if self.context.token_count > self.max_tokens:
                self.condition(False)
                return

        # Generate punctuation.
        await self.observe(
            self.context.mask_dist(self.context.lm.masks.END_PUNCTUATION), True
        )
        await self.sample(self.context.next_token())

        self.sentence_count += 1

        # If we reach the sentence target, force the model to end the paragraph.
        if self.sentence_count >= self.n_sentences_target:
            await self.observe(self.context.mask_dist(set([self.EOS_TOKEN])), True)
            await self.sample(self.context.next_token())
            self.finish()
            return
