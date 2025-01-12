from manim import *

class EqnDance(Scene):
    def create(self):
        eqn=MathTex("a^2")
        print(eqn)
        self.add(eqn)
