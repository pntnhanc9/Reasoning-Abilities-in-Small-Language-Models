import asyncio
import string

import nltk

from disciple.base_models import BaseModel
from disciple.masks import EOSMask
from disciple.masks import NextTokenMask
from disciple.masks import PunctuationMask
from disciple.masks import TokenLengthMask
from evaluations.model_registry import ModelRegistry


@ModelRegistry.register(
    task_type="length_constraints",
    task_id=None,
    dataset="examples",
    prompt="Please generate a sentence with between 80 and 100 characters. Include whitespace into your character count.",
    reasoning="The task is to generate a sentence with a character count in a certain range. We can generate token-by-token, using masks to enforce the length constraint. We can provide a hint about the remaining length at each step. Once the minimum length is reached, we can allow the model to end the sentence or keep generating. However, once the maximum length is reached, we need to force the model to generate EOS. (When this occurs, we shouldn't force it to generate punctuation, since this would exceed the character limit.) Since each token contributes to the length constraint, this seems like a good choice for a 'token' step granularity.",
)
class ExampleModelLengthConstraints(BaseModel):
    """Generates a sentence with between 80 and 100 characters, including whitespace."""

    def __init__(
        self,
        context,
        max_tokens: int = 32,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.min_length = 80
        self.max_length = 100

    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a token that fits within the remaining length.
        Only allow EOS if we've reached the minimum length.

        Step granularity: token

        End condition: EOS token is sampled or token limit is reached.
        """

        # Provide a hint about the remaining length.
        if len(self) < self.min_length:
            await self.hint(
                f"We need to generate at least {self.min_length - len(self)} more characters."
            )
        else:
            await self.hint(f"There are {self.max_length - len(self)} characters left.")

        # Sample a token that fits within the remaining length.
        async with TokenLengthMask(
            self,
            max_chars=self.max_length - len(self),
            allow_eos=len(self) >= self.min_length,
        ):
            token = await self.next_token()

        # If EOS was generated, end the generation.
        # NOTE: We don't have to sample/observe punctuation; we assume that the model will have already generated punctuation by the time EOS is generated.
        if int(token) == self.EOS_TOKEN_ID:
            await self.end()
            return
        # If we've reached the maximum length, force EOS to be generated.
        # NOTE: We avoid manually sampling punctuation here as this would create an off-by-one error in the length constraint.
        # While it's possible here that punctuation will not have already been generated, particles *with* punctuation will have higher weight than particles that were truncated, so they will be preferred by SMC.
        elif len(self) >= self.max_length:
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the length constraints."""
        return self.min_length <= len(text) <= self.max_length


@ModelRegistry.register(
    task_type="target_words",
    task_id=None,
    dataset="examples",
    prompt="Please generate a sentence containing at least 3 of the following words: 'apple', 'banana', 'cherry', 'date', 'grape', 'kiwi'.",
    reasoning="The task is to generate a sentence containing at least 3 words from a list of target words. We can generate word-by-word, keeping track of the remaining target words and provide a hint about the remaining target words at each step. The main challenge here is that the ordering of the target words is not fixed, and we want the model to be able to generate them in any order. Since we only want to resample after satisfying a constraint, this seems like a good choice for a 'phrase' granularity, where each step produces one of the target words. The final step extends the generation freely until EOS. To make sure that we generate exactly one target word per step, it's extremely important that we `return` right after sampling a target word.",
)
class ExampleModelTargetWords(BaseModel):
    def __init__(
        self,
        context,
        max_tokens: int = 32,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.target_words = set(["apple", "banana", "cherry", "date", "grape", "kiwi"])
        self.generated_target_words = set([])
        self.min_target_words = 3

        # Maximum number of words before generating one of the targets
        self.max_words_per_phrase = 40

    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate one of the target words.
        At each step, keep sampling words until a target word is generated.
        As soon as a target word is generated, `return` to complete the step.
        On the final step (after at least 3 target words have been generated), extend the sentence unconstrained until EOS.

        Step granularity: phrase

        End condition: At least 3 target words have been generated or token limit is reached.
        """
        # Each step samples words until a target word is generated.
        if len(self.generated_target_words) < self.min_target_words:

            await self.hint(
                f"The following target words are remaining: {self.target_words.difference(self.generated_target_words)}."
            )

            for _ in range(1, self.max_words_per_phrase + 1):
                word = await self.next_word()

                # Remove the generated word from the target set.
                if word.strip().lower() in self.target_words:
                    self.generated_target_words.add(word.strip().lower())

                    # IMPORTANT: Return after generating the target word to complete the step.
                    return

                # Enforce token limit
                if self.context.token_count > self.max_tokens:
                    self.condition(False)
                    await self.end()
                    return

            # If we reach the word limit without generating a target word, reject
            self.condition(False)

        # If all target words have been generated, extend until EOS
        else:
            await self.hint(
                f"At least {self.min_target_words} have already been generated."
            )

            await self.extend()
            await self.end()

    async def check(self, text: str) -> bool:
        """Check that the generated text contains at least 3 target words."""
        words = set(nltk.word_tokenize(text.lower()))
        return len(words.intersection(self.target_words)) >= self.min_target_words


@ModelRegistry.register(
    task_type="sent_01",
    task_id=205,
    dataset="collie",
    prompt="Please generate a sentence with exactly 82 characters. Include whitespace into your character count.",
    reasoning="The task is to generate a sentence with a fixed number of characters. We can generate token-by-token, using masks to enforce the length constraint. We can provide a hint about the remaining length at each step, and force the model to end the sentence once the target length is reached. Since each token contributes to the length constraint, this seems like a good choice for a 'token' step granularity.",
)
class CollieModelSent01(BaseModel):
    """Generates a sentence with exactly 82 characters, including whitespace."""

    def __init__(
        self,
        context,
        max_tokens: int = 32,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.target_length = 82

    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a token that fits within the remaining length.
        Once we've reached the target length, end the generation.

        Step granularity: token

        End condition: Target length is reached or token limit is reached.
        """
        remaining_length = self.target_length - len(self)

        # Provide a hint about the remaining length.
        await self.hint(f"There are {remaining_length} characters left.")

        # Sample a token that fits within the remaining length.
        async with TokenLengthMask(
            self,
            max_chars=remaining_length,
            allow_eos=(len(self) >= self.target_length),
        ):
            await self.next_token()

        # Once we've reached the maximum length, end the generation.
        # NOTE: We avoid manually sampling punctuation here as this would create an off-by-one error in the length constraint.
        if len(self) >= self.target_length:
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the length constraints."""
        return len(text) == self.target_length


@ModelRegistry.register(
    task_type="sent_02",
    task_id=243,
    dataset="collie",
    prompt="Please generate a sentence:\n1) with exactly 11 words;\n2) with the 4th, 8th, 11th words to be 'Series', 'and', '4' respectively.",
    reasoning="The task is to generate a sentence with a fixed number of words and specific words at certain positions. We can generate word-by-word, keeping track of the index, and force the model to sample the target words at the specified positions. However, we need to be careful about step granularity; we want each step to satisfy a single constraint. This seems like a good choice for a 'phrase' granularity, where each step generates exactly *one* of the target words. To make sure that we generate exactly one target word per step, it's extremely important that we `return` right after sampling a target word.",
)
class CollieModelSent02(BaseModel):
    """Generates a sentence with exactly 11 words, where the 4th, 8th, and 11th words are fixed to be 'Series', 'and', '4' respectively."""

    def __init__(
        self,
        context,
        max_tokens: int = 32,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.word_idx = 1
        self.max_words = 11
        self.target_words = {
            4: "Series",
            8: "and",
            11: "4",
        }

    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate one of the target words.
        At each step, keep sampling words until a we reach a target index.
        For the target index, force the model to sample the target word.
        As soon as a target word is generated, `return` to complete the step.
        Once we hit max words, force punctuation and end.

        Step granularity: phrase

        End condition: Word index exceeds max_words or token limit is reached.
        """

        # Provide a hint about the remaining words.
        await self.hint(
            f"The following words still need to be generated: {[word for i, word in self.target_words.items() if i >= self.word_idx]}."
        )

        # Sample words until we hit the next target word.
        for i in range(self.word_idx, self.max_words + 1):

            # If the current word index corresponds to a target word, generate that word.
            if self.word_idx in self.target_words:
                word = self.target_words[self.word_idx]
                await self.extend_with(word)
                self.word_idx += 1

                # IMPORTANT: Return after generating the target word to complete the step.
                return

            # Otherwise, sample a word unconstrained.
            else:
                await self.next_word()
                self.word_idx += 1

        # Once max_words is reached, generate punctuation and end.
        if self.word_idx > self.max_words:
            async with PunctuationMask(self):
                await self.next_token()
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        words = [w for w in nltk.word_tokenize(text) if w not in string.punctuation]
        if len(words) != self.max_words:
            return False
        for idx, word in self.target_words.items():
            if words[idx - 1].lower() != word.lower():
                return False
        return True


@ModelRegistry.register(
    task_type="sent_03",
    task_id=341,
    dataset="collie",
    prompt="Please generate a sentence:\n1) with at least 9 words;\n2) with all words having at most 7 characters.",
    reasoning="The task is to generate a sentence with a minimum number of words, where each word has a maximum number of characters. We can generate word-by-word, keeping track of the word index, and enforce the word length constraint. We can provide a hint about the number of words remaining at each step. Once the minimum number of words is reached, we can allow the model to end the sentence. Since the word length constraint applies to each individual word, this seems like a good choice for a 'word' granularity, where we generate one word at each step.",
)
class CollieModelSent03(BaseModel):
    """Generates a sentence with at least 9 words, where each word has at most 7 characters."""

    def __init__(
        self,
        context,
        max_tokens: int = 32,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.min_words = 9
        self.max_chars_per_word = 7

        self.word_idx = 1

    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a word, keeping track of the word index.
        After the min_words limit is reached, allow (but do not force) end punctuation to be generated.
        Once end punctuation is generated, end the generation.
        We use max_chars to limit the length of each word.

        Step granularity: word

        End condition: Word index exceeds min_words or token limit is reached.
        """

        # Provide a hint about the number of words remaining.
        await self.hint(
            f"There are at least {self.min_words - self.word_idx} words left to generate."
        )

        # Sample a word with a maximum length of max_chars
        word = await self.next_word(
            max_chars=self.max_chars_per_word,
        )
        self.word_idx += 1

        # If we've reached the min_words limit, allow the model to end the sentence
        if self.word_idx > self.min_words:
            if await self.sample(PunctuationMask(self)):
                await self.next_token()
                await self.end()
                return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        words = [w for w in nltk.word_tokenize(text) if w not in string.punctuation]
        if len(words) < self.min_words:
            return False
        for word in words:
            if len(word) > self.max_chars_per_word:
                return False
        return True


@ModelRegistry.register(
    task_type="sent_04",
    task_id=370,
    dataset="collie",
    prompt="Please generate a sentence containing the word 'have', 'rising', 'the'.",
    reasoning="The task is to generate a sentence containing specific words. We can generate word-by-word, keeping track of the remaining target words and provide a hint about the remaining target words at each step. Once all target words are generated, we can extend the sentence unconstrained until end punctuation is generated. The main challenge here is that the ordering of the target words is not fixed, and we want the model to be able to generate them in any order. Since we only want to resample after satisfying a constraint, this seems like a good choice for a 'phrase' granularity, where we generate multiple words at each step until a target word is generated. To make sure that we generate exactly one target word per step, it's extremely important that we `return` right after sampling a target word.",
)
class CollieModelSent04(BaseModel):
    """Generates a sentence containing the words 'have', 'rising', and 'the'."""

    def __init__(self, context, max_tokens: int = 32):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.target_words = set(["have", "rising", "the"])

        # Maximum number of words before generating one of the targets
        self.max_words_per_phrase = 40

    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate one of the target words.
        At each step, keep sampling words until one of the target words is generated.
        As soon as a target word is generated, `return` to complete the step.
        On the final step (once all target words have been generated), extend the sentence unconstrained until EOS.

        Step granularity: phrase

        End condition: All target words have been generated or token limit is reached.
        """
        # If any target words are still present, keep sampling words until one is generated.
        if len(self.target_words) > 0:

            await self.hint(
                f"The following target words are remaining: {self.target_words}."
            )

            for _ in range(1, self.max_words_per_phrase + 1):
                word = await self.next_word()

                # Remove the generated word from the target set.
                if word.strip().lower() in self.target_words:
                    self.target_words.remove(word.strip().lower())

                    # IMPORTANT: Return after generating the target word to complete the step.
                    return

                # Enforce token limit
                if self.context.token_count > self.max_tokens:
                    self.condition(False)
                    await self.end()
                    return

            # If we reach the word limit without generating a target word, reject
            self.condition(False)

        # If all target words have been generated, extend until EOS
        else:
            await self.extend()
            await self.end()

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        words = [w for w in nltk.word_tokenize(text) if w not in string.punctuation]
        for word in self.target_words:
            if word.lower() not in words:
                return False
        return True


@ModelRegistry.register(
    task_type="para_01",
    task_id=465,
    dataset="collie",
    prompt="Please generate a paragraph with all sentences having the 1st word to be 'The'.",
    reasoning="The task is to generate a paragraph where each sentence starts with a specific word. We can generate sentence-by-sentence, sampling the target word first and then generating the rest of the sentence. We can provide a hint about the target word at each step. Once the target word is generated, we can extend the sentence unconstrained until end punctuation is generated. The main challenge here is that the target word is fixed for each sentence, and we want the model to generate it first. This seems like a good choice for a 'sentence' granularity, where we generate one sentence at each step.",
)
class CollieModelPara01(BaseModel):
    """Generates a paragraph where each sentence starts with the word 'The'."""

    def __init__(
        self,
        context,
        max_tokens: int = 128,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.target_word = "The"

    @classmethod
    def prior_prompt(cls):
        return "Write a paragraph that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a sentence starting with the target word.
        Optionally end the paragraph after each sentence.

        Step granularity: sentence

        End condition: EOS token is sampled or token limit is reached.
        """

        # Generate the sentence
        await self.extend(start=self.target_word, stop=[".", "!", "?"])

        # Allow the model to optionally end the paragraph
        if await self.sample(EOSMask(self)):
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        sentences = nltk.sent_tokenize(text.lower())
        for sentence in sentences:
            if not sentence.startswith(self.target_word.lower()):
                return False
        return True


@ModelRegistry.register(
    task_type="para_02",
    task_id=473,
    dataset="collie",
    prompt="Please generate a paragraph:\n1) with exactly 3 sentences;\n2) not containing the word 'be';\n3) not containing the word 'this';\n4) not containing the word 'is'.",
    reasoning="The task is to generate a paragraph while excluding specific words. The main challenge is how to avoid disallowed words. While we could try to do this via token masking, we can also rely on the proposal (which knows the task instructions) to try to avoid these words. Therefore, a simpler strategy is to just check each word against the constraints and reject if we encounter a disallowed word. Then, we can generate sentence-by-sentence, where each step contains a loop that samples words until the sentence is complete. Once the target number of sentences is reached, we can end the generation.",
)
class CollieModelPara02(BaseModel):
    """Generates a paragraph with exactly 3 sentences, excluding the words 'be', 'this', and 'is'."""

    def __init__(
        self,
        context,
        max_tokens: int = 128,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.n_sentences_target = 3
        self.disallowed_words = set([word.lower() for word in ["be", "this", "is"]])
        self.max_words_per_sentence = 100

        self.sentence_count = 0

    @classmethod
    def prior_prompt(cls):
        return "Write a paragraph that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a sentence word-by-word.
        If a disallowed word is generated, reject the sentence.
        Optionally end the paragraph after each sentence.

        Step granularity: sentence

        End condition: n_sentences_target is reached or token limit is reached.
        """

        end_punctuation = None

        # Provide a hint about the remaining sentences.
        await self.hint(
            f"There are {self.n_sentences_target - self.sentence_count} sentences left to generate."
        )

        # Sample the sentence word-by-word
        for _ in range(self.max_words_per_sentence):
            word = await self.next_word()

            # If the word is disallowed, reject
            if word.lower() in self.disallowed_words:
                self.condition(False)
                return

            # If we reach the end of the sentence, break
            if await self.sample(PunctuationMask(self)):
                end_punctuation = await self.next_token()
                break

        # Reject the sentence if we reach the word limit without end punctuation
        if not end_punctuation:
            self.condition(False)
            return

        self.sentence_count += 1

        # If we reach n_sentences_target, end generation
        if self.sentence_count >= self.n_sentences_target:
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        sentences = nltk.sent_tokenize(text.lower())
        if len(sentences) != self.n_sentences_target:
            return False
        for sentence in sentences:
            words = [
                w for w in nltk.word_tokenize(sentence) if w not in string.punctuation
            ]
            for word in words:
                if word.lower() in self.disallowed_words:
                    return False
        return True


@ModelRegistry.register(
    task_type="para_03",
    task_id=567,
    dataset="collie",
    prompt="Please generate a paragraph:\n1) with exactly 4 sentences;\n2) with all sentences having at least 12 words;\n3) with all sentences having at most 20 words.",
    reasoning="The task is to generate a paragraph with a fixed number of sentences, where each sentence has a minimum and maximum number of words. Since the constraints apply to each individual sentence, this seems like a good choice for a 'sentence' granularity, where we generate one sentence at each step. Once the target number of sentences is reached, we can end the generation. The main challenge here is generating sentences with the right number of words. To do this, we can sample word-by-word and allow end punctuation to be generated after the minimum number of words is met. If a sentence reaches the word maximum, we could either force punctuation to be generated, or just reject the sentence. In this case, if the sentence wasn't finished by the time the word limit was reached, adding punctuation is probably going to be unnatural, so we can just reject it. We can also use hints to make the proposal model aware of both the remaining number of words in the sentence, and the remaining number of sentences in the paragraph.",
)
class CollieModelPara03(BaseModel):
    """Generates a paragraph with exactly 4 sentences, where each sentence has between 12 and 20 words."""

    def __init__(
        self,
        context,
        max_tokens: int = 128,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.n_sentences_target = 4
        self.min_words_per_sentence = 12
        self.max_words_per_sentence = 20

        self.sentence_count = 0

    @classmethod
    def prior_prompt(cls):
        return "Write a paragraph that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a sentence word-by-word.
        After min_words_per_sentence have been generated, allow (but do not force) end punctuation to be generated.
        Once n_sentences_target is reached, end the generation.

        Step granularity: sentence

        End condition: n_sentences_target is reached or token limit is reached.
        """

        end_punctuation = None

        # Sample the sentence word-by-word, allowing end punctuation if min_words_per_sentence have been generated
        for word_idx in range(1, self.max_words_per_sentence + 1):

            # Provide a hint about the remaining words in the sentence.
            hint_text = f"This is sentence {self.sentence_count + 1} / {self.n_sentences_target}."
            if word_idx < self.min_words_per_sentence:
                hint_text += f" There are at least {self.min_words_per_sentence - word_idx} words left to generate."
            else:
                hint_text += f" There are at most {self.max_words_per_sentence - word_idx} words left to generate."
            await self.hint(hint_text)

            word = await self.next_word()

            # End the sentence as soon as end punctuation is generated
            if word_idx >= self.min_words_per_sentence:
                if await self.sample(PunctuationMask(self)):
                    end_punctuation = await self.next_token()
                    break

        # Reject the sentence if we reach the word limit without end punctuation
        if not end_punctuation:
            self.condition(False)

        self.sentence_count += 1

        # If we reach n_sentences_target, end generation
        if self.sentence_count >= self.n_sentences_target:
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        sentences = nltk.sent_tokenize(text.lower())
        if len(sentences) != self.n_sentences_target:
            return False
        for sentence in sentences:
            words = [
                w for w in nltk.word_tokenize(sentence) if w not in string.punctuation
            ]
            if (
                len(words) < self.min_words_per_sentence
                or len(words) > self.max_words_per_sentence
            ):
                return False
        return True


@ModelRegistry.register(
    task_type="para_04",
    task_id=660,
    dataset="collie",
    prompt="Please generate a paragraph:\n1) with at least 3 sentences;\n2) with all sentences having at least 21 words.",
    reasoning="The task is to generate a paragraph with a minimum number of sentences, where each sentence has a minimum number of words. We can generate sentence-by-sentence, sampling words until the sentence is complete. We can provide a hint about the remaining words in the sentence. Once the minimum number of sentences is reached, we can allow the model to end the paragraph. The main challenge here is that we want to allow end punctuation to be generated after a minimum number of words, but not force it. Therefore, we can sample word-by-word and allow end punctuation to be generated after a minimum number of words.",
)
class CollieModelPara04(BaseModel):
    """Generates a paragraph with at least 3 sentences, where each sentence has at least 21 words."""

    def __init__(
        self,
        context,
        max_tokens: int = 128,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.min_sentences = 3
        self.min_words_per_sentence = 21
        self.max_words_per_sentence = 100

        self.sentence_count = 0

    @classmethod
    def prior_prompt(cls):
        return "Write a paragraph that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample a sentence word-by-word.
        After min_words_per_sentence have been generated, allow (but do not force) end punctuation to be generated.
        After min_sentences is reached, allow (but do not force) the paragraph to end.

        Step granularity: sentence

        End condition: EOS token is sampled or token limit is reached.
        """

        end_punctuation = None

        # Sample the sentence word-by-word, allowing end punctuation if min_words_per_sentence have been generated
        for word_idx in range(1, self.max_words_per_sentence + 1):

            # Provide a hint about the remaining words in the sentence.
            hint_text = (
                f"This is sentence {self.sentence_count + 1} / {self.min_sentences}."
            )
            hint_text += f" This sentence contains {word_idx} / {self.min_words_per_sentence} words."
            await self.hint(hint_text)

            word = await self.next_word()

            # End the sentence as soon as end punctuation is generated
            if word_idx >= self.min_words_per_sentence:
                if await self.sample(PunctuationMask(self)):
                    end_punctuation = await self.next_token()
                    break

        # Reject the sentence if we reach the word limit without end punctuation
        if not end_punctuation:
            self.condition(False)
            return

        self.sentence_count += 1

        # If we reach min_sentences, optionally end the paragraph
        if self.sentence_count >= self.min_sentences:
            if await self.sample(EOSMask(self)):
                await self.end()
                return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        sentences = nltk.sent_tokenize(text.lower())
        if len(sentences) < self.min_sentences:
            return False
        for sentence in sentences:
            words = [
                w for w in nltk.word_tokenize(sentence) if w not in string.punctuation
            ]
            if len(words) < self.min_words_per_sentence:
                return False
        return True


@ModelRegistry.register(
    task_type="para_05",
    task_id=680,
    dataset="collie",
    prompt="Please generate a paragraph:\n1) with exactly 3 sentences;\n2) with sentences having the last word to be 'convention', 'president', 'Wisconsin' respectively.",
    reasoning="The task is to generate a paragraph with a fixed number of sentences, where each sentence ends with a specific word. We can generate sentence-by-sentence, sampling words until the target word is generated. The main challenge is making sure that the sentence ends with the right target word. We can use hints to encourage the proposal model to use the right target word for each sentence. To make sure that this is the last word of the sentence, as soon as we generate a target word, we can force the model to generate punctuation. Since we want to resample after satisfying a constraint, this seems like a good choice for a 'sentence' granularity, where we generate one sentence at each step.",
)
class CollieModelPara05(BaseModel):
    """Generates a paragraph with exactly 3 sentences, where each sentence ends with 'convention', 'president', 'Wisconsin'."""

    def __init__(
        self,
        context,
        max_tokens: int = 128,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.n_sentences_target = 3
        self.max_words_per_sentence = 100
        self.target_last_words = [
            word.lower() for word in ["convention", "president", "Wisconsin"]
        ]

        self.sentence_count = 0

    @classmethod
    def prior_prompt(cls):
        return "Write a paragraph that is grammatically correct and makes sense."

    async def step(self):
        """
        Generation strategy:
        At each step, sample word-by-word with no punctuation.
        Once the target word is generated, add punctuation to the end of the sentence.
        Once n_sentences_target is reached, end the generation.

        Step granularity: sentence

        End condition: n_sentences_target is reached or token limit is reached.
        """

        last_word_reached = False

        # Sample the sentence word-by-word
        for word_idx in range(1, self.max_words_per_sentence + 1):

            # Provide a hint about the remaining words in the sentence.
            hint_text = f"This is sentence {self.sentence_count + 1} / {self.n_sentences_target}."
            hint_text += f" This sentence needs to end with the word '{self.target_last_words[self.sentence_count]}'."
            await self.hint(hint_text)

            # Sample the next word, deferring punctuation until the target word is reached
            word = await self.next_word()

            # If the word is the target word, end the sentence
            if word.strip().lower() == self.target_last_words[self.sentence_count]:
                last_word_reached = True
                break

        # Reject the sentence if we reach the word limit without the target word
        if not last_word_reached:
            self.condition(False)

        # Now add punctuation to the end of the sentence
        # Since next_word() doesn't generate punctuation, we need to do this manually
        async with PunctuationMask(self):
            await self.next_token()

        self.sentence_count += 1

        # If we reach n_sentences_target, end generation
        if self.sentence_count >= self.n_sentences_target:
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        """Check that the generated text satisfies the word constraints."""
        sentences = nltk.sent_tokenize(text.lower())
        if len(sentences) != self.n_sentences_target:
            return False
        for idx, sentence in enumerate(sentences):
            words = [
                w for w in nltk.word_tokenize(sentence) if w not in string.punctuation
            ]
            if words[-1].lower() != self.target_last_words[idx]:
                return False
        return True
