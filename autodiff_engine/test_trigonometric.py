"""
Test script for trigonometric and exponential functions in autodiff.
"""
from value import Value
import math


def test_exponential_functions() -> None:
    print("--- Exponential Functions ---")
    x = Value(2.0)
    y = x.exp()
    y.backward()
    print(f"y = exp({x.value}) = {y.value}")
    print(f"dy/dx = {x.grad}")
    print(f"Expected: exp(2) = {math.exp(2)}")
    print(f"Expected gradient: exp(2) = {math.exp(2)}")
    y.visualize('graph_exp', format='svg')
    print("Graph saved as graph_exp.svg\n")


def test_logarithm() -> None:
    print("--- Logarithm ---")
    x = Value(3.0)
    y = x.log()
    y.backward()
    print(f"y = log({x.value}) = {y.value}")
    print(f"dy/dx = {x.grad}")
    print(f"Expected: log(3) = {math.log(3)}")
    print(f"Expected gradient: 1/3 = {1/3}")
    y.visualize('graph_log', format='svg')
    print("Graph saved as graph_log.svg\n")


def test_trigonometric_functions() -> None:
    print("--- Trigonometric Functions ---")
    x = Value(math.pi / 4)  # 45 degrees
    y1 = x.sin()
    y2 = x.cos()
    y3 = x.tan()
    
    y1.backward()
    print(f"sin(π/4) = {y1.value}")
    print(f"d(sin)/dx = {x.grad}")
    print(f"Expected: sin(π/4) = {math.sin(math.pi/4)}")
    print(f"Expected gradient: cos(π/4) = {math.cos(math.pi/4)}")
    y1.visualize('graph_sin', format='svg')
    print("Graph saved as graph_sin.svg\n")


def test_hyperbolic_functions() -> None:
    print("--- Hyperbolic Functions ---")
    x = Value(1.0)
    y1 = x.sinh()
    y2 = x.cosh()
    y3 = x.tanh()
    
    y1.backward()
    print(f"sinh(1) = {y1.value}")
    print(f"d(sinh)/dx = {x.grad}")
    print(f"Expected: sinh(1) = {math.sinh(1)}")
    print(f"Expected gradient: cosh(1) = {math.cosh(1)}")
    y1.visualize('graph_sinh', format='svg')
    print("Graph saved as graph_sinh.svg\n")


def test_complex_expression() -> None:
    print("--- Complex Expression ---")
    x = Value(0.5)
    y = Value(1.0)
    
    # Complex expression: exp(sin(x)) * cosh(y) + log(x + y)
    expr = x.sin().exp() * y.cosh() + (x + y).log()
    expr.backward()
    
    print(f"expr = exp(sin(x)) * cosh(y) + log(x + y)")
    print(f"x = {x.value}, y = {y.value}")
    print(f"expr = {expr.value}")
    print(f"dexpr/dx = {x.grad}")
    print(f"dexpr/dy = {y.grad}")
    
    # Manual calculation for verification
    expected = (math.exp(math.sin(0.5)) * math.cosh(1.0) + math.log(0.5 + 1.0))
    print(f"Expected value: {expected}")
    
    expr.visualize('graph_complex_trig', format='svg')
    print("Graph saved as graph_complex_trig.svg\n")


def test_chain_rule_with_trig() -> None:
    print("--- Chain Rule with Trigonometric ---")
    x = Value(1.0)
    y = x.sin().exp() * x.cos()
    y.backward()
    
    print(f"y = exp(sin(x)) * cos(x)")
    print(f"x = {x.value}")
    print(f"y = {y.value}")
    print(f"dy/dx = {x.grad}")
    
    # Manual calculation
    expected_value = math.exp(math.sin(1.0)) * math.cos(1.0)
    expected_grad = (math.exp(math.sin(1.0)) * math.cos(1.0) * math.cos(1.0) - 
                    math.exp(math.sin(1.0)) * math.sin(1.0))
    print(f"Expected value: {expected_value}")
    print(f"Expected gradient: {expected_grad}")
    
    y.visualize('graph_chain_trig', format='svg')
    print("Graph saved as graph_chain_trig.svg\n")


def main() -> None:
    test_exponential_functions()
    test_logarithm()
    test_trigonometric_functions()
    test_hyperbolic_functions()
    test_complex_expression()
    test_chain_rule_with_trig()


if __name__ == "__main__":
    main() 