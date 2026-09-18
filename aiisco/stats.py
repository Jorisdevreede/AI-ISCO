"""Rank statistics for comparing two sets of scores of the same skills."""

BANDS = 5


def mean(values):
    """Arithmetic mean of a non-empty sequence."""
    return sum(values) / len(values)


def share(flags):
    """Fraction of a non-empty sequence of booleans that are true."""
    return sum(flags) / len(flags)


def tied_runs(values, order):
    """Yield (first, last) index pairs into `order` that share the same value."""
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        yield i, j
        i = j + 1


def ranks(values):
    """1-based ranks of `values`, tied values sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for first, last in tied_runs(values, order):
        shared = (first + last) / 2 + 1
        for k in range(first, last + 1):
            out[order[k]] = shared
    return out


def spearman(xs, ys):
    """Spearman rank correlation, 0.0 when either side has no variation at all."""
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def band(score_1_to_10):
    """Rubric band 0-4 for a 1-10 score (1-2, 3-4, 5-6, 7-8, 9-10)."""
    return min(4, max(0, int((score_1_to_10 - 1) // 2)))


def top_band(probs):
    """The rubric band a probability distribution puts the most weight on."""
    return max(range(BANDS), key=lambda i: probs[i])
