"""A deterministic synthetic story corpus.

This exists so the whole pipeline — tokenizer training, packing, training,
sampling — is runnable and testable with no network and no GPU. CI runs it, and
so does anyone cloning the repo before they get a Kaggle session.

It is **not** a substitute for TinyStories. It is generated from a small
context-free grammar with a few hundred distinct surface forms, so a model can
fit it almost exactly; a low loss here says the training loop works, not that
the model is good. The M0 gate is defined on real TinyStories and can only be
cleared on real data.

Why a grammar rather than random tokens: random tokens have no learnable
structure, so a broken model and a working one produce the same loss curve
(flat at ``ln(vocab_size)``). With a grammar, loss falling well below the
unigram entropy is positive evidence that the model is actually learning
sequence structure, which makes this usable as an integration test.
"""

from __future__ import annotations

import random

NAMES = ["Lily", "Tom", "Anna", "Ben", "Mia", "Sam", "Zoe", "Max", "Ivy", "Leo"]
ANIMALS = ["cat", "dog", "bird", "frog", "fox", "bear", "duck", "mouse"]
PLACES = ["park", "forest", "garden", "beach", "house", "school", "river", "hill"]
OBJECTS = ["ball", "book", "kite", "box", "hat", "cup", "flower", "stone"]
ADJECTIVES = ["little", "happy", "red", "big", "soft", "bright", "kind", "tiny"]
FEELINGS = ["happy", "sad", "excited", "proud", "surprised", "sleepy"]

TEMPLATES = [
    "One day {name} went to the {place}. {name} found a {adj} {obj}. "
    "The {obj} was very {adj2}. {name} was {feeling}.",
    "{name} had a {adj} {animal}. The {animal} liked to play in the {place}. "
    "One day the {animal} found a {obj}. {name} was {feeling}.",
    "There was a {adj} {animal} who lived near the {place}. "
    "Every day the {animal} looked for a {obj}. Then {name} came and helped. "
    "The {animal} was {feeling}.",
    "{name} and {name2} went to the {place} together. They took a {adj} {obj} with them. "
    "They played all day. Then they went home and were {feeling}.",
    "The {adj} {animal} wanted a {obj}. It walked to the {place} to look for one. "
    "{name} gave the {animal} a {obj}. Now the {animal} is {feeling}.",
]


def generate_corpus(n_stories: int, seed: int = 0) -> list[str]:
    """Generate ``n_stories`` stories. Deterministic in ``seed``."""
    rng = random.Random(seed)
    stories = []
    for _ in range(n_stories):
        template = rng.choice(TEMPLATES)
        name, name2 = rng.sample(NAMES, 2)
        stories.append(
            template.format(
                name=name,
                name2=name2,
                animal=rng.choice(ANIMALS),
                place=rng.choice(PLACES),
                obj=rng.choice(OBJECTS),
                adj=rng.choice(ADJECTIVES),
                adj2=rng.choice(ADJECTIVES),
                feeling=rng.choice(FEELINGS),
            )
        )
    return stories
