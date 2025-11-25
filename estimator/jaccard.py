from typing import Iterable

def score(set1: set | Iterable, set2: set | Iterable, sig_digits=3) -> float:
    """
    Calculate the Jaccard score between two sets.

    Parameters:
    set1 (set): The first set
    set2 (set): The second set

    Returns:
    float: The Jaccard score
    """
    set1, set2 = set(set1), set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return round(intersection / union, sig_digits)
