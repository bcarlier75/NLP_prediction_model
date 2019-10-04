import os
import re


def strip_formatting(string: str):
    """Lower the string and separate words from punctuation."""
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string


def test_file_validity(path: str):
    """Check path existence and if it leads to a file."""
    if os.path.exists(path):
        if os.path.isfile(path):
            return 1
    return 0


def print_results(n: str, p: float, r: float):
    """Print model prediction metrics."""
    print("Number of samples\t" + str(n))
    print("Precision@{}\t\t{:.3f}".format(1, p))
    print("Recall@{}\t\t{:.3f}".format(1, r))
    print()
