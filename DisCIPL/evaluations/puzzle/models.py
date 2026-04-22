import asyncio
import datetime
import re
import string
from typing import Tuple

import nltk

from disciple.base_models import BaseModel
from disciple.masks import EOSMask
from disciple.masks import NewLineMask
from disciple.masks import NextTokenMask
from disciple.masks import PunctuationMask
from disciple.masks import TokenLengthMask
from evaluations.model_registry import ModelRegistry
from evaluations.puzzle.tasks import *


@ModelRegistry.register(
    task_type=SquareWordPoem().task_types[0],
    task_id=None,
    dataset="puzzle",
    prompt=SquareWordPoem().prompt,
)
class SquareWordPoemModel(BaseModel):
    """Generates a poem with N lines, where each line has exactly N words."""

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
        self.N = 8  # Number of lines and words per line
        self.line_i = 1

    @classmethod
    def prior_prompt(cls):
        return "Write a poem."

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate a line of the poem.
        To generate a line, sample word-by-word N times.
        After generating a line, sample a newline token to move to the next line.

        Step granularity: line

        End conditions:
        1. N lines are generated.
        2. The token limit is reached.
        """
        for _ in range(self.N):
            await self.next_word()
        async with NewLineMask(self, n=1):
            await self.next_token()
            self.line_i += 1
        if self.line_i > self.N:
            await self.end()
        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        lines = text.strip().split("\n")
        if len(lines) != self.N:
            return False
        for line in lines:
            words = line.split()
            if len(words) != self.N:
                return False
        return True


@ModelRegistry.register(
    task_type=GrantProposal().task_types[0],
    task_id=None,
    dataset="puzzle",
    prompt=GrantProposal().prompt,
)
class GrantProposalModel(BaseModel):
    "Generates an abstract for a grant proposal on elephant ecology and conservation. The abstract starts with 'Abstract:'. It is between 75 and 100 words and excludes a list of forbidden words."

    def __init__(
        self,
        context,
        max_tokens: int = 512,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.forbidden_words = set(
            [
                "conservation",
                "sustainability",
                "environment",
                "ecology",
                "wildlife",
                "africa",
                "asia",
                "society",
                "community",
                "biodiversity",
                "endangered",
                "threatened",
                "species",
                "habitat",
                "poaching",
                "science",
                "research",
            ]
        )
        self.min_words = 75
        self.max_words = 100
        self.word_count = 0
        self.header = False

    @classmethod
    def prior_prompt(cls):
        return "Write an abstract for a grant proposal on elephant ecology and conservation."

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate a word for the abstract.
        On the first step, we generate the header, "Abstract:".
        On subsequent steps, we sample a word and check if it is a forbidden word.
        If the word is forbidden, reject.
        After each word, check if the model wants to sample punctuation.
        If the minimum word count is reached, additionally allow the model to sample EOS.
        If the maximum word count is reached, reject.

        Step granularity: word

        End conditions:
        1. The model samples EOS.
        2. The maximum word count is reached.
        3. The token limit is reached.
        """

        # Generate the title first.
        if not self.header:
            await self.extend_with("Abstract:")
            self.header = True
            return

        hint_text = f"The current length is {self.word_count} words."
        if self.word_count < self.min_words:
            hint_text += (
                f" We need at least {self.min_words - self.word_count} more words."
            )
        else:
            hint_text += f" There are only {self.max_words - self.word_count} words left before we hit the limit!"
        await self.hint(hint_text)

        # Sample a word.
        word = await self.next_word()
        self.word_count += 1

        # Check if the sentence contains any forbidden words.
        if word.strip().lower() in self.forbidden_words:
            self.condition(False)
            return

        # Optionally, sample punctuation (but do not end, since the abstract will contain multiple sentences).
        if await self.sample(PunctuationMask(self)):
            await self.next_token()

        # If the minimum word count has been reached, allow the model to sample EOS.
        if self.min_words <= self.word_count < self.max_words:
            if await self.sample(EOSMask(self)):
                await self.end()
                return
        # If the maximum word count has been reached, reject.
        elif self.word_count >= self.max_words:
            self.condition(False)
            await self.end()
            return

        # Enforce token limit.
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

    async def check(self, text: str) -> bool:
        text = text.strip()

        # Check for the header
        if not text.startswith("Abstract:"):
            return False

        # Extract abstract
        abstract = text[len("Abstract:") :].strip()
        words = abstract.lower().split()

        # Check word count
        word_count = len(words)
        if word_count < self.min_words or word_count > self.max_words:
            return False

        # Check for forbidden words first
        for word in words:
            if word in self.forbidden_words:
                return False

        return True


@ModelRegistry.register(
    task_type=IngredientsList().task_types[0],
    task_id=None,
    dataset="puzzle",
    prompt=IngredientsList().prompt,
)
class IngredientsListModel(BaseModel):
    """Writes an ingredients list for chocolate chip brownies with at most 7 ingredients costing less than $18.00 total. The list is in bullet point format starting with "Ingredients:". Each ingredient is listed on a separate line with the price given in USD."""

    def __init__(
        self,
        context,
        max_tokens: int = 256,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.max_ingredients = 7
        self.max_cost = 18.00

        self.line_i = 0
        self.total_cost = 0.0

    @classmethod
    def prior_prompt(cls):
        return "Write an ingredients list for chocolate chip brownies."

    def extract_cost(self, text: str) -> float:
        match = re.search(r"\$(\d+(?:\.\d+)?)", text)
        if not match:
            return None
        try:
            cost = float(match.group(1))
        except ValueError:
            return None

        cost = round(cost, 2)
        return cost

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate an ingredient for the list.
        To generate an ingredient, sample the ingredient name and price.
        We can use a hint to inform the model of the remaining budget.
        If the cost of the ingredient exceeds the maximum cost, reject.
        After generating an ingredient, check if the model wants to sample EOS.
        If the maximum number of ingredients is reached, force the model to sample EOS.

        Step granularity: line

        End conditions:
        1. The model samples EOS.
        2. The maximum number of ingredients is reached.
        3. The cost limit is reached.
        4. The token limit is reached
        """
        # The first step generates the header "Ingredients:"
        if self.line_i == 0:
            await self.extend_with("Ingredients:\n")
            self.line_i += 1
            return

        # Provide a hint about the remaining budget
        await self.hint(
            f"The remaining budget is ${self.max_cost - self.total_cost:.2f}."
        )

        # Generate the ingredient
        # Ensure the line starts with a hyphen and ends with a newline
        # Set allow_eos=True to allow the model to sample EOS
        ingredient, eos = await self.extend(start="-", stop=["\n"], allow_eos=True)

        # Extract the cost of the ingredient
        cost = self.extract_cost(ingredient)
        if cost is None:
            self.condition(False)
            return

        # Update the running total cost
        self.total_cost += cost
        if self.total_cost > self.max_cost:
            self.condition(False)
            return

        # If the model sampled EOS on this step, end the generation
        if eos:
            await self.end()
            return
        # If the maximum number of ingredients is reached, force the model to sample EOS
        elif self.line_i >= self.max_ingredients:
            await self.end()
            return

        # Enforce token limit
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

        self.line_i += 1

    async def check(self, text: str) -> bool:
        lines = text.strip().split("\n")
        if lines[0] != "Ingredients:":
            return False
        if len(lines) > self.max_ingredients + 1:
            return False

        total_cost = 0
        for line in lines[1:]:
            if not line:
                continue
            if not line.startswith("-"):
                return False
            cost = self.extract_cost(line)
            if cost is None:
                return False
            total_cost += cost

        if total_cost > self.max_cost:
            return False

        return True


@ModelRegistry.register(
    task_type=TripItinerary().task_types[0],
    task_id=None,
    dataset="puzzle",
    prompt=TripItinerary().prompt,
)
class TripItineraryModel(BaseModel):
    """Generates a three-day trip itinerary with at least 4 activities per day. Each day should start with "Day N:" and end with a double newline. Each activity should start with a time range in 24-hour format in square brackets (e.g., "[11:00-13:00]") and end with a newline. The itinerary should include at least 9 hours of free time each day for rest."""

    def __init__(
        self,
        context,
        max_tokens: int = 512,
    ):
        super().__init__(
            context=context,
            max_tokens=max_tokens,
        )

        # Task-specific variables
        self.n_days = 3
        self.n_activities = 4
        self.day_i = 1
        self.activity_i = 0

        self.min_free_time = 9

    @classmethod
    def prior_prompt(cls):
        return "Write a day-by-day itinerary for a 3-day trip to Singapore."

    def extract_time_range(
        self, text: str
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        match = re.search(r"\[\d{2}:\d{2}-\d{2}:\d{2}\]", text)
        if not match:
            return None
        time_range = match.group(0)
        # Remove square brackets and split into start and end times
        time_range_clean = time_range.strip("[]")
        try:
            start_str, end_str = time_range_clean.split("-")
            start_time = datetime.datetime.strptime(start_str, "%H:%M")
            end_time = datetime.datetime.strptime(end_str, "%H:%M")
        except ValueError:
            return None
        return start_time, end_time

    def is_complete(self) -> bool:
        return self.day_i >= self.n_days and self.activity_i >= self.n_activities

    async def step(self):
        """
        Generation strategy:
        Each step is going to generate a line in the itinerary consisting of a time range and an activity.
        Each line should start with a time range in 24-hour format (e.g., "[11:00-13:00]") and end with a single or double newline.
        If the line ends with a double newline, check to make sure there are at least 4 activities for the day and move to the next day.
        After enough activities are generated on the final day, allow the model to sample EOS to end the itinerary.

        Step granularity: line

        End conditions:
        1. The model ends the itinerary after generating enough activities for each day.
        2. The model prematurely ends the itinerary without generating enough activities.
        3. The free time limit is reached.
        4. The token limit is reached.
        """
        # Generate the header for the day.
        if self.activity_i == 0:
            await self.extend_with(f"Day {self.day_i}:\n")
            self.activity_i += (
                1  # IMPORTANT: Increment activity_i to avoid infinite loop.
            )
            self.free_time = 24
            return

        # Generate the activity.
        # Ensure the line starts with a time range in 24-hour format and ends with a newline.
        # On the final day, once the model generates enough activities, allow it to sample EOS.
        activity, eos = await self.extend(
            start="[", stop=["\n"], allow_eos=self.is_complete()
        )

        # Extract the time range of the activity.
        time_range = self.extract_time_range(activity)
        if time_range is None:
            self.condition(False)
            return

        # Check if there is enough free time for the day.
        start_time, end_time = time_range
        activity_duration = (end_time - start_time).seconds / 3600
        self.free_time -= activity_duration
        if self.free_time < self.min_free_time:
            self.condition(False)
            return

        # If the model sampled EOS on this step, end the generation.
        if eos:
            await self.end()
            return

        # If the model generated a double newline, move to the next day.
        if activity.endswith("\n\n"):

            # If there aren't enough activities for the day, reject.
            if self.activity_i < self.n_activities:
                self.condition(False)
                return

            # If this is the final day, force the model to end.
            if self.day_i == self.n_days:
                await self.end()
                return

            # Move to the next day.
            self.day_i += 1
            self.activity_i = 0
            return

        # Enforce token limit.
        if self.context.token_count > self.max_tokens:
            self.condition(False)
            await self.end()
            return

        # Move to the next activity.
        self.activity_i += 1

    async def check(self, text: str) -> bool:
        days = text.strip().split("\n\n")
        if len(days) != 3:
            return False

        for day in days:
            lines = day.split("\n")
            if not lines[0].startswith("Day"):
                return False

            activities = lines[1:]
            if len(activities) < 4:
                return False

            free_time = 24
            for activity in activities:
                time_range = self.extract_time_range(activity)
                if time_range is None:
                    return False
                start_time, end_time = time_range
                activity_duration = (end_time - start_time).seconds / 3600

                # Ensure the activity duration is non-negative.
                if activity_duration < 0:
                    return False
                free_time -= activity_duration

            # Ensure there is enough free time for the day.
            if free_time < self.min_free_time:
                return False

        return True
