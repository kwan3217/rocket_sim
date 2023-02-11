"""
Some of the shots are in manim. Maybe eventually all of the shots will be.
"""

from manim import *
import numpy as np

background="#e0e0ff"
eqn_color='#1a5fb4'


class PEG0(Scene):
    def construct(self):
        eqn = MathTex("a^2")
        print(eqn)
        self.add(eqn)


