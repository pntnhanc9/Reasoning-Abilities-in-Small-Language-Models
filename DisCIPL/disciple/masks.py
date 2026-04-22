from llamppl.distributions import Distribution
from llamppl.distributions.lmcontext import LMTokenMask
from llamppl.distributions.lmcontext import NullMask


class NextTokenMask(Distribution):
    """A context manager for masking tokens during generation.

    Provides the same interface as LMTokenMask but with the added ability to function as an async context manager.
    When used with the `with` syntax, the mask is observed, so the next token generated will be sampled from the mask.
    Note that the mask only applies to the next token generated, even if there are multiple tokens generated within the context manager.

    Args:
        model (BaseModel): The model to apply the mask to.
        include (set): The set of tokens to include in the mask. All other tokens will be excluded from generation.
    """

    def __init__(self, model, include: set):
        self.model = model
        self.include = include
        self.mask = LMTokenMask(model.context, include)

    def invert(self):
        """Inverts the mask so that all tokens in the mask are excluded from generation."""
        self.include = self.model.context.lm.masks.ALL_TOKENS - self.include
        self.mask = LMTokenMask(self.model.context, self.include)
        return self

    async def sample(self):
        return await self.mask.sample()

    async def log_prob(self, v):
        return await self.mask.log_prob(v)

    async def __aenter__(self):
        await self.model.observe(self.mask, True)

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class PunctuationMask(NextTokenMask):
    """A context manager for generating end-of-sentence punctuation tokens.

    END_PUNCTUATION = [".", "!", "?"]

    Args:
        model (BaseModel): The model to apply the mask to.
    """

    def __init__(self, model):
        super().__init__(model, model.context.lm.masks.END_PUNCTUATION)


class NewLineMask(NextTokenMask):
    """A context manager for generating new line tokens.

    Args:
        model (BaseModel): The model to apply the mask to.
        n (int): The number of newlines to generate. If None, the mask will include all tokens containing a newline character.
    """

    def __init__(self, model, n: int = 1):
        if n is None:
            include = set(
                [i for i, v in enumerate(model.context.lm.str_vocab) if "\n" in v]
            )
        else:
            include = set([model.context.lm.str_vocab.index("\n" * n)])
        super().__init__(model, include)


class EOSMask(NextTokenMask):
    """A context manager for generating the EOS token.

    Args:
        model (BaseModel): The model to apply the mask to.
    """

    def __init__(self, model):
        super().__init__(model, model.context.lm.masks.EOS)


class TokenLengthMask(NextTokenMask):
    """A context manager for limiting the number of characters generated.

    NOTE: Special tokens like EOS are treated as having length 0.

    Args:
        model (BaseModel): The model to apply the mask to.
        min_chars (int): The minimum number of characters to generate.
        max_chars (int): The maximum number of characters to generate.
        allow_eos (bool): Whether to allow the EOS token to be generated.
    """

    def __init__(
        self,
        model,
        min_chars: int = None,
        max_chars: int = None,
        allow_eos: bool = True,
    ):
        mask = model.context.lm.masks.token_length_mask(min_chars, max_chars)
        if len(mask) == 0:
            print(
                f"WARNING: TokenLengthMask is empty for min_chars={min_chars}, max_chars={max_chars}, allow_eos={allow_eos}. Setting allow_eos=True."
            )

        if allow_eos or len(mask) == 0:
            mask = mask.union(model.context.lm.masks.EOS)
        else:
            mask = mask.difference(model.context.lm.masks.EOS)

        super().__init__(model, mask)
