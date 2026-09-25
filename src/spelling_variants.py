"""
Common Roman-Urdu spelling variant clusters.

Roman-Urdu has no standard orthography, so the same word gets written
many ways. Each list below is a cluster of spellings that all mean the
same thing -- used for (a) light normalization at inference time, and
(b) generating augmented training examples.

NOTE: not exhaustive -- covers high-frequency, sentiment-relevant words
found during error analysis / canary testing. Extend as you find more gaps.
"""

import re
import random

VARIANT_CLUSTERS = [
    ["hai", "hy", "h", "hae", "hea"],
    ["hain", "hen", "hn"],
    ["nahi", "nahin", "nai", "ni", "nhi"],
    ["bohat", "bahut", "bht", "boht", "bahot"],
    ["acha", "achha", "atcha", "acha"],
    ["bura", "bure", "buri", "burah"],
    ["pasand", "pasnd"],
    ["kharab", "kharaab", "khrab"],
    ["zabardast", "zbrdst"],
]

# canonical form -> all variants (including itself)
_CANONICAL_MAP = {}
for cluster in VARIANT_CLUSTERS:
    canonical = cluster[0]
    for word in cluster:
        _CANONICAL_MAP[word] = cluster


def normalize_spelling(text: str) -> str:
    """Light-touch: map known variants to their canonical spelling.
    Does NOT lowercase/strip punctuation -- safe to use right before
    feeding text to a transformer at inference time."""
    words = text.split()
    normalized = []
    for w in words:
        key = w.lower()
        if key in _CANONICAL_MAP:
            normalized.append(_CANONICAL_MAP[key][0])
        else:
            normalized.append(w)
    return " ".join(normalized)


def generate_variants(text: str, n: int = 2, seed: int = None) -> list:
    """Generate up to n augmented copies of `text`, each with one
    variant-prone word swapped for a different spelling from its cluster.
    Returns [] if the sentence contains no known variant-prone words."""
    rng = random.Random(seed)
    words = text.split()
    swappable_positions = [i for i, w in enumerate(words) if w.lower() in _CANONICAL_MAP]

    if not swappable_positions:
        return []

    results = []
    for _ in range(n):
        new_words = words.copy()
        pos = rng.choice(swappable_positions)
        cluster = _CANONICAL_MAP[words[pos].lower()]
        alt_options = [w for w in cluster if w != words[pos].lower()]
        if alt_options:
            new_words[pos] = rng.choice(alt_options)
            candidate = " ".join(new_words)
            if candidate != text:
                results.append(candidate)
    return list(set(results))  # dedupe