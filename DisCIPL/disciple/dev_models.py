from disciple.base_models import BaseModel


class DebugModel(BaseModel):
    """BaseModel subclass for debugging."""

    async def step(self):
        self.finish()


class UnconditionalSamplingModel(BaseModel):
    """Unconditional sampling model."""

    @classmethod
    def prior_prompt(cls):
        return "Please generate some text."

    async def step(self):
        """Each step samples an entire completion."""

        while True:
            # Sample token from language model
            token = await self.sample(self.context.next_token())

            # On EOS, finish
            if token == self.EOS_TOKEN_ID:
                self.finish()
                return

            # If we've run out of tokens, fail
            if self.context.token_count > self.max_tokens:
                self.condition(False)
                return

    async def check(self, text: str) -> bool:
        pass
