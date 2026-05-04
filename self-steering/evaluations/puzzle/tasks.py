import datetime
import re
import string
from typing import Tuple

import nltk

from evaluations.dataset import Task


def load_all_tasks():
    return [
        SquareWordPoem(),
        GrantProposal(),
        IngredientsList(),
        TripItinerary(),
    ]


class SquareWordPoem(Task):
    def __init__(self, N: int = 8):
        self.N = N
        super().__init__(
            task_id=None,
            prompt=f"Write a poem with {self.N} lines, where each line has exactly {self.N} words. (For our purposes, a word is defined as a sequence of characters separated by whitespace.)",
            evaluators={
                "valid": lambda text: SquareWordPoem.valid(text, self.N),
            },
            task_types=["square_poem"],
        )

    @staticmethod
    def valid(text, N):
        lines = text.strip().split("\n")
        if len(lines) != N:
            print(f"X Number of lines: {len(lines)}")
            return False
        for line in lines:
            words = line.split()
            if len(words) != N:
                print(f"X Number of words: {len(words)}")
                print(words)
                return False
        return True


class SquareCharacterPoem(Task):
    def __init__(self):
        super().__init__(
            task_id=None,
            prompt="Your goal is to write a square poem. That means the poem should be a single stanza with the same number of characters on each line, so that the poem appears as a square.",
            evaluators={
                "valid": SquareCharacterPoem.valid,
            },
            task_types=["square_poem"],
        )

    @staticmethod
    def valid(text):
        lines = text.split("\n")
        if len(set(len(line) for line in lines)) != 1:
            return False
        return True


class RectangularCharacterPoem(Task):
    def __init__(self):
        super().__init__(
            task_id=None,
            prompt='Write a poem with 3 "rectangular" stanzas. Within each stanza, the number of characters on every line should be equal, so that the stanza appears as a rectangle.',
            evaluators={
                "valid": RectangularCharacterPoem.valid,
            },
            examples=[self.example],
            task_types=["rectangular_poem"],
        )

    @property
    def example(self):
        return (
            "roses are reddish,\n"
            "and violets r blue\n"
            "\n"
            "this poem is square\n"
            "so it's just like u\n"
            "\n"
            "i hope you like it"
        )

    @property
    def negative_example(self):
        return (
            "roses are red,\n"
            "and violets are blue\n"
            "\n"
            "this poem is not square\n"
            "so it's just like u"
        )

    @staticmethod
    def valid(text):
        stanzas = text.split("\n\n")
        if len(stanzas) != 3:
            return False

        for stanza in stanzas:
            lines = stanza.split("\n")
            line_lengths = [len(line) for line in lines]
            if len(set(line_lengths)) != 1:
                return False

        return True


class ElephantInThePoem(Task):
    forbidden_words = [
        "elephant",
        "trunk",
        "tusk",
        "ivory",
        "jungle",
        "big",
        "huge",
        "majestic",
        "the",
        "and",
        "of",
    ]

    def __init__(self):
        super().__init__(
            task_id=None,
            prompt=f"Write a poem about elephants without using any of the following words: {', '.join(self.forbidden_words)}. (For our purposes, a word is defined as a sequence of characters separated by whitespace.)",
            evaluators={
                "valid": ElephantInThePoem.valid,
            },
            task_types=["elephant_poem"],
        )

    @property
    def example(self):
        return "The noble pachyderm is large and gray,\nIt roams the savannah all day."

    @property
    def negative_example(self):
        return "The noble elephant is large and gray,\nIt roams the savannah all day."

    @staticmethod
    def valid(text):
        text = text.strip().lower()
        words = text.split()
        for word in words:
            if word in ElephantInThePoem.forbidden_words:
                return False
        return True


class GrantProposal(Task):
    forbidden_words = [
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

    min_words = 75
    max_words = 100

    def __init__(self):
        super().__init__(
            task_id=None,
            prompt=f"Write an abstract for a grant proposal on elephant ecology and conservation. Your response should start with \"Abstract: <YOUR ABSTRACT>\". The abstract must be between {self.min_words}-{self.max_words} words. You are not allowed to use any of the following terms: {', '.join(self.forbidden_words)}. (For our purposes, a word is defined as a sequence of characters separated by whitespace.)",
            evaluators={
                "valid": GrantProposal.valid,
            },
            task_types=["grant_proposal"],
        )

    @staticmethod
    def valid(text):
        text = text.strip()

        if not text.startswith("Abstract:"):
            print("X Abstract section does not start with 'Abstract:'")
            return False

        # Extract abstract
        abstract = text[len("Abstract:") :].strip()
        words = abstract.lower().split()

        # Check word count
        word_count = len(words)
        if word_count < GrantProposal.min_words or word_count > GrantProposal.max_words:
            print(f"X Word count: {word_count}")
            return False

        # Check for forbidden words first
        for word in words:
            if word in GrantProposal.forbidden_words:
                print(f"X Forbidden word: {word}")
                return False

        return True


class IngredientsList(Task):
    def __init__(self):
        super().__init__(
            task_id=None,
            prompt='Please write an ingredients list for chocolate chip brownies with at most 7 ingredients costing less than $18.00 total. The list should be in dashed bullet point format starting with "Ingredients:". Each ingredient should be listed on a separate line with the price given in USD.',
            evaluators={
                "valid": IngredientsList.valid,
            },
            task_types=["ingredients_list"],
        )

    @property
    def example(self):
        return (
            "Ingredients:\n"
            "- flour $2.00\n"
            "- sugar $1.00\n"
            "- eggs $3.00\n"
            "- butter $4.00\n"
            "- chocolate chips $5.00\n"
            "- vanilla extract $3.00"
        )

    @property
    def negative_example(self):
        return (
            "Ingredients:\n"
            "- flour $2.00\n"
            "- sugar $1.00\n"
            "- eggs $3.00\n"
            "- butter $4.00\n"
            "- chocolate chips $5.00\n"
            "- vanilla extract $3.00\n"
            "- baking powder $1.00"
        )

    @staticmethod
    def valid(text):
        lines = text.strip().split("\n")
        if lines[0] != "Ingredients:":
            print("First line is not 'Ingredients:'")
            return False
        if len(lines) > 8:
            print(f"Too many lines: {len(lines)}")
            return False

        total_cost = 0
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("-"):
                print("Line does not start with a dash")
                return False
            match = re.search(r"\$(\d+(?:\.\d+)?)", line)
            if not match:
                print(f"No price found: {line}")
                return False
            try:
                cost = float(match.group(1))
            except ValueError:
                print(f"Invalid price: {match.group(1)}")
                return False
            total_cost += cost

        if total_cost > 18.00:
            print(f"Total cost exceeds $18.00: {total_cost}")
            return False

        return True


class TripItinerary(Task):
    def __init__(self):
        super().__init__(
            task_id=None,
            prompt='I\'m planning a 3-day trip to Singapore. Please write me a detailed day-by-day itinerary that includes at least four activities per day. The itinerary should start with "Day 1:" and end with "Day 3:", with a blank line between each day. Each activity should be listed on a separate line starting with a time range in 24-hour format in square brackets (for example, "[11:00-13:00] Visit the Gardens by the Bay"). Make sure to leave at least 9 hours of free time each day for rest.',
            evaluators={
                "valid": TripItinerary.valid,
            },
            task_types=["trip_itinerary"],
        )

    @property
    def example(self):
        return (
            "Day 1:\n"
            "[09:00-11:00] Breakfast at a local hawker center\n"
            "[11:00-13:00] Visit the Gardens by the Bay\n"
            "[13:00-14:00] Lunch at Satay by the Bay\n"
            "[14:00-16:00] Explore Marina Bay Sands\n"
            "\n"
            "Day 2:\n"
            "[09:00-11:00] Breakfast at Chinatown Food Street\n"
            "[11:00-13:00] Visit the National Museum of Singapore\n"
            "[13:00-14:00] Lunch at Maxwell Food Centre\n"
            "[14:00-16:00] Explore Little India\n"
            "\n"
            "Day 3:\n"
            "[09:00-11:00] Breakfast at Tiong Bahru Market\n"
            "[11:00-13:00] Visit the Singapore Zoo\n"
            "[13:00-14:00] Lunch at Ah Meng Restaurant\n"
            "[14:00-16:00] Explore Sentosa Island"
        )

    @property
    def negative_example(self):
        return (
            "Day 1:\n"
            "[09:00-11:00] Breakfast at a local hawker center\n"
            "\n"
            "Day 2:\n"
            "[09:00-11:00] Breakfast at Chinatown Food Street\n"
            "[11:00-13:00] Visit the National Museum of Singapore\n"
            "[13:00-14:00] Lunch at Maxwell Food Centre\n"
            "[14:00-16:00] Explore Little India\n"
            "\n"
            "Day 3:\n"
            "[09:00-11:00] Breakfast at Tiong Bahru Market\n"
            "[11:00-13:00] Visit the Singapore Zoo\n"
            "[13:00-14:00] Lunch at Ah Meng Restaurant\n"
        )

    @staticmethod
    def extract_time_range(text: str) -> Tuple[datetime.datetime, datetime.datetime]:
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

    @staticmethod
    def valid(text):
        days = re.split(r"\n\n+", text.strip())
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
                time_range = TripItinerary.extract_time_range(activity)
                if not time_range:
                    return False
                start_time, end_time = time_range
                activity_duration = (end_time - start_time).seconds / 3600
                free_time -= activity_duration
                if activity_duration < 0:
                    return False
            if free_time < 9:
                return False

        return True
