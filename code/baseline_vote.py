"""Training-free baseline: vote on parsed final answers; ties use first occurrence."""
from collections import Counter
from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        # Unparseable answers do not count as agreement.
        counts = Counter(answer for answer in answers if answer is not None)
        if not counts:
            choices.append(0)
            continue
        # max keeps the first candidate if the vote counts are tied.
        choice = max(range(4), key=lambda i: counts[answers[i]] if answers[i] is not None else -1)
        choices.append(choice)
    return choices
