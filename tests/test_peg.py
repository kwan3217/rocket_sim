"""
Foo all the Bars
"""

import pytest
import peg

def test_fly():
    test_rocket=peg.PEG()
    test_rocket.calc_tau()
    test_rocket.calculate_steering()
    test_rocket.fly()
