import numpy as np
from manim import *

class ChainRuleMovingTangents(Scene):
    def construct(self):
        # 1. Setup Title and Axes
        title = Tex("The Chain Rule:", " $(f \\circ g)'(x) = f'(g(x)) \\cdot g'(x)$").to_edge(UP)
        title[0].set_color(WHITE)
        title[1].set_color(YELLOW)
        self.play(Write(title), run_time=2)

        axes_g = Axes(
            x_range=[-4, 4, 1], y_range=[0, 5, 1], x_length=6, y_length=3.5,
            axis_config={"include_tip": False}
        ).to_edge(DL, buff=1)
        axes_f = Axes(
            x_range=[0, 5, 1], y_range=[-2, 2, 1], x_length=6, y_length=3.5,
            axis_config={"include_tip": False}
        ).to_edge(DR, buff=1)
        
        g_axis_label = axes_g.get_graph_label(axes_g.plot(lambda x:0), "x")
        u_axis_label_g = axes_g.get_graph_label(axes_g.plot(lambda x:0), "u=g(x)", direction=UL, x_val=-4)
        u_axis_label_f = axes_f.get_graph_label(axes_f.plot(lambda x:0), "u")
        z_axis_label_f = axes_f.get_graph_label(axes_f.plot(lambda x:0), "z=f(u)", direction=UL, x_val=0)
        
        self.play(Create(VGroup(axes_g, axes_f)), Write(VGroup(g_axis_label, u_axis_label_g, u_axis_label_f, z_axis_label_f)), run_time=2)
        self.wait(0.5)

        # 2. Define and plot functions
        g = lambda x: 0.25 * x**2 + 1
        f = lambda u: np.sin(u)
        
        graph_g = axes_g.plot(g, x_range=[-4, 4], color=BLUE)
        graph_f = axes_f.plot(f, x_range=[0, 5], color=YELLOW)
        
        graph_g_label = axes_g.get_graph_label(graph_g, "g(x)", x_val=-3.5, direction=UP)
        graph_f_label = axes_f.get_graph_label(graph_f, "f(u)", x_val=PI/2, direction=UP)

        self.play(Create(graph_g), Create(graph_f), Write(graph_g_label), Write(graph_f_label), run_time=1.5)
        self.wait(1)

        # 3. Create trackers and moving elements
        x_tracker = ValueTracker(2)
        
        g_x_range = [-4, 4]
        f_u_range = [0, 5]

        # Moving dots and tangents
        dot_g = always_redraw(lambda: Dot(axes_g.input_to_graph_point(x_tracker.get_value(), graph_g), color=BLUE))
        tangent_g = always_redraw(lambda: TangentLine(graph_g, alpha=(x_tracker.get_value() - g_x_range[0]) / (g_x_range[1] - g_x_range[0]), length=3, color=BLUE_D))
        
        u_val = lambda: axes_g.p2d(dot_g.get_center())[1]
        
        dot_f = always_redraw(lambda: Dot(axes_f.input_to_graph_point(u_val(), graph_f), color=YELLOW))
        tangent_f = always_redraw(lambda: TangentLine(graph_f, alpha=(u_val() - f_u_range[0]) / (f_u_range[1] - f_u_range[0]), length=3, color=YELLOW_D))

        # 4. Add labels for slopes and the final product
        g_prime_label = Tex("$g'(x) = $", font_size=36).to_corner(UL).shift(DOWN*0.5)
        f_prime_label = Tex("$f'(g(x)) = $", font_size=36).next_to(g_prime_label, DOWN, aligned_edge=LEFT)
        
        g_prime_val_text = always_redraw(lambda: Tex(f"{axes_g.slope_of_tangent(x=x_tracker.get_value(), graph=graph_g):.2f}", font_size=36).next_to(g_prime_label, RIGHT))
        f_prime_val_text = always_redraw(lambda: Tex(f"{axes_f.slope_of_tangent(x=u_val(), graph=graph_f):.2f}", font_size=36).next_to(f_prime_label, RIGHT))

        result_text = Tex("$(f \\circ g)'(x) = $", font_size=40).next_to(f_prime_label, DOWN, buff=0.5, aligned_edge=LEFT)
        
        def get_result_mobject():
            g_prime = axes_g.slope_of_tangent(x=x_tracker.get_value(), graph=graph_g)
            f_prime = axes_f.slope_of_tangent(x=u_val(), graph=graph_f)
            return Tex(f"${f_prime:.2f} \\cdot {g_prime:.2f} = {f_prime * g_prime:.2f}$", font_size=40).next_to(result_text, RIGHT)

        result_val = always_redraw(get_result_mobject)
        
        labels = VGroup(g_prime_label, f_prime_label, g_prime_val_text, f_prime_val_text, result_text, result_val)
        self.play(Create(VGroup(dot_g, tangent_g, dot_f, tangent_f)), Write(labels), run_time=2)
        self.wait(1)

        # 5. Animate the tracker
        self.play(x_tracker.animate.set_value(-3), run_time=6, rate_func=linear)
        self.wait(0.5)
        self.play(x_tracker.animate.set_value(3.5), run_time=7, rate_func=linear)
        self.wait(1)