from typing import List


# Returns jaccard index returning the closeness of the two sets.
# For custom equality, override the default classes.
def jaccard(produced: List[any], correct: List[any]) -> float:
    p = set(produced)
    c = set(correct)
    pnc = p.intersection(c)
    puc = p.union(c)

    return len(pnc) / len(puc)


# set-precision computation
def precision(produced: List[any], correct: List[any]) -> float:
    p = set(produced)
    c = set(correct)
    pnc = p.intersection(c)

    # Avoid division by zero errors
    if len(produced) == 0:
        return -1

    # TP / (TP + FP)
    return len(pnc) / len(produced)


# set-recall computation
def recall(produced: List[any], correct: List[any]) -> float:
    p = set(produced)
    c = set(correct)
    pnc = p.intersection(c)

    # TP / (TP + FN)
    return len(pnc) / len(correct)
