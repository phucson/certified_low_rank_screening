"""Effective tail support s_eta(sigma) by family: the localisation boundary.

Localisation (and hence the banded fallback) is cheap only when a small upper
tail carries essentially all the spectral weight.
"""
import sys
sys.path.insert(0, "..")
from certproj import FAMILIES, effective_tail_support

if __name__ == "__main__":
    print(f"{'sigma':<16}{'s_0.99':>10}")
    for name, g in FAMILIES.items():
        print(f"{name:<16}{effective_tail_support(g, eta=0.99):>10.3f}")
