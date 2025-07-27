"""
Test script for autodiff Value and operations, including visualization.
"""
from value import Value


def test_basic_operations() -> None:
    print("--- Basic Operations ---")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    print(f"z = x * y + x = {z.value}")
    print(f"dz/dx = {x.grad}")
    print(f"dz/dy = {y.grad}")
    z.visualize('graph_basic', format='svg')
    print("Graph saved as graph_basic.svg\n")


def test_complex_expression() -> None:
    print("--- Complex Expression ---")
    a = Value(1.5)
    b = Value(-2.0)
    c = Value(0.5)
    expr = ((a + b) * (c - a) / b) ** 2
    expr.backward()
    print(f"expr = ((a + b) * (c - a) / b) ** 2 = {expr.value}")
    print(f"dexpr/da = {a.grad}")
    print(f"dexpr/db = {b.grad}")
    print(f"dexpr/dc = {c.grad}")
    expr.visualize('graph_complex', format='svg')
    print("Graph saved as graph_complex.svg\n")


def test_chain_rule() -> None:
    print("--- Chain Rule ---")
    x = Value(2.0)
    y = (x * 3.0 + 1.0) ** 3
    y.backward()
    print(f"y = (x * 3 + 1) ** 3 = {y.value}")
    print(f"dy/dx = {x.grad}")
    y.visualize('graph_chain', format='svg')
    print("Graph saved as graph_chain.svg\n")


def main() -> None:
    test_basic_operations()
    test_complex_expression()
    test_chain_rule()


if __name__ == "__main__":
    main() 