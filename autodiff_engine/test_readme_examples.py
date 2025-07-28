#!/usr/bin/env python3
"""
Test file containing all examples from the README to ensure they work correctly.
This file validates that all code examples in the documentation are functional.

Note: This implementation uses reverse-mode automatic differentiation (backpropagation),
not forward mode as originally described in the README.
"""

import math
from value import Value

def test_basic_usage() -> None:
    """Test the basic usage example from README."""
    print("=== Testing Basic Usage Example ===")
    
    # Create variables (with optional labels for better visualization)
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    
    # Note: Intermediate results can be labeled manually for better visualization
    
    # Build computational graph
    z = x * y + x**2
    
    # Compute gradients
    z.backward()
    
    print(f"z = {z.value}")
    print(f"∂z/∂x = {x.grad}")
    print(f"∂z/∂y = {y.grad}")
    
    # Verify expected results
    expected_z = 2.0 * 3.0 + 2.0**2  # 6 + 4 = 10
    expected_dz_dx = 3.0 + 2 * 2.0   # 3 + 4 = 7
    expected_dz_dy = 2.0              # 2
    
    assert abs(z.value - expected_z) < 1e-10, f"Expected z = {expected_z}, got {z.value}"
    assert abs(x.grad - expected_dz_dx) < 1e-10, f"Expected ∂z/∂x = {expected_dz_dx}, got {x.grad}"
    assert abs(y.grad - expected_dz_dy) < 1e-10, f"Expected ∂z/∂y = {expected_dz_dy}, got {y.grad}"
    print("✅ Basic usage test passed!")

def test_trigonometric_functions() -> None:
    """Test the trigonometric functions example from README."""
    print("\n=== Testing Trigonometric Functions Example ===")
    
    x = Value(math.pi / 4, label='x')
    y = x.sin() + x.cos()
    y.backward()
    
    print(f"y = {y.value}")
    print(f"∂y/∂x = {x.grad}")
    
    # Verify expected results
    expected_y = math.sin(math.pi / 4) + math.cos(math.pi / 4)  # √2/2 + √2/2 = √2
    expected_dy_dx = math.cos(math.pi / 4) - math.sin(math.pi / 4)  # √2/2 - √2/2 = 0
    
    assert abs(y.value - expected_y) < 1e-10, f"Expected y = {expected_y}, got {y.value}"
    assert abs(x.grad - expected_dy_dx) < 1e-10, f"Expected ∂y/∂x = {expected_dy_dx}, got {x.grad}"
    print("✅ Trigonometric functions test passed!")

def test_complex_expressions() -> None:
    """Test the complex expressions example from README."""
    print("\n=== Testing Complex Expressions Example ===")
    
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = Value(1.0, label='c')
    
    # Complex expression: (a^2 + b) * sin(c)
    result = (a**2 + b) * c.sin()
    result.backward()
    
    print(f"Result = {result.value}")
    print(f"∂result/∂a = {a.grad}")
    print(f"∂result/∂b = {b.grad}")
    print(f"∂result/∂c = {c.grad}")
    
    # Verify expected results
    expected_result = (2.0**2 + 3.0) * math.sin(1.0)  # (4 + 3) * sin(1) = 7 * sin(1)
    expected_dresult_da = 2 * 2.0 * math.sin(1.0)     # 2a * sin(c) = 4 * sin(1)
    expected_dresult_db = math.sin(1.0)                # sin(c) = sin(1)
    expected_dresult_dc = (2.0**2 + 3.0) * math.cos(1.0)  # (a² + b) * cos(c) = 7 * cos(1)
    
    assert abs(result.value - expected_result) < 1e-10, f"Expected result = {expected_result}, got {result.value}"
    assert abs(a.grad - expected_dresult_da) < 1e-10, f"Expected ∂result/∂a = {expected_dresult_da}, got {a.grad}"
    assert abs(b.grad - expected_dresult_db) < 1e-10, f"Expected ∂result/∂b = {expected_dresult_db}, got {b.grad}"
    assert abs(c.grad - expected_dresult_dc) < 1e-10, f"Expected ∂result/∂c = {expected_dresult_dc}, got {c.grad}"
    print("✅ Complex expressions test passed!")

def test_labeling_for_visualization() -> None:
    """Test the labeling for visualization example from README."""
    print("\n=== Testing Labeling for Visualization Example ===")
    
    # Create labeled input values
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")
    angle = Value(0.5, label="θ")
    
    # For intermediate results, you can manually assign labels
    z = x * y
    z.label = "z = x*y"  # Label intermediate result
    
    result = z + angle.sin()
    result.label = "result = z + sin(θ)"  # Label final result
    
    # The labels will appear in the graph visualization
    result.visualize('readme_labeled_graph')
    print("✅ Labeling for visualization test passed!")
    print("   Graph saved as 'readme_labeled_graph.svg'")

def test_example_1_linear_function() -> None:
    """Test Example 1: Simple Linear Function from README."""
    print("\n=== Testing Example 1: Simple Linear Function ===")
    
    x = Value(2.0, label='x')
    y = 3 * x + 1
    y.backward()
    
    print(f"f(x) = 3x + 1")
    print(f"f({x.value}) = {y.value}")
    print(f"f'({x.value}) = {x.grad}")  # Should be 3
    
    # Verify expected results
    expected_y = 3 * 2.0 + 1  # 7
    expected_dy_dx = 3         # 3
    
    assert abs(y.value - expected_y) < 1e-10, f"Expected f(2) = {expected_y}, got {y.value}"
    assert abs(x.grad - expected_dy_dx) < 1e-10, f"Expected f'(2) = {expected_dy_dx}, got {x.grad}"
    print("✅ Linear function test passed!")

def test_example_2_quadratic_function() -> None:
    """Test Example 2: Quadratic Function from README."""
    print("\n=== Testing Example 2: Quadratic Function ===")
    
    x = Value(3.0, label='x')
    y = x**2 + 2*x + 1
    y.backward()
    
    print(f"f(x) = x² + 2x + 1")
    print(f"f({x.value}) = {y.value}")
    print(f"f'({x.value}) = {x.grad}")  # Should be 2x + 2 = 8
    
    # Verify expected results
    expected_y = 3.0**2 + 2*3.0 + 1  # 9 + 6 + 1 = 16
    expected_dy_dx = 2*3.0 + 2        # 6 + 2 = 8
    
    assert abs(y.value - expected_y) < 1e-10, f"Expected f(3) = {expected_y}, got {y.value}"
    assert abs(x.grad - expected_dy_dx) < 1e-10, f"Expected f'(3) = {expected_dy_dx}, got {x.grad}"
    print("✅ Quadratic function test passed!")

def test_example_3_trigonometric_function() -> None:
    """Test Example 3: Trigonometric Function from README."""
    print("\n=== Testing Example 3: Trigonometric Function ===")
    
    x = Value(math.pi / 6, label='x')  # 30 degrees
    y = x.sin()
    y.backward()
    
    print(f"f(x) = sin(x)")
    print(f"f({x.value}) = {y.value}")
    print(f"f'({x.value}) = {x.grad}")  # Should be cos(x)
    
    # Verify expected results
    expected_y = math.sin(math.pi / 6)  # sin(30°) = 0.5
    expected_dy_dx = math.cos(math.pi / 6)  # cos(30°) = √3/2 ≈ 0.866
    
    assert abs(y.value - expected_y) < 1e-10, f"Expected f(π/6) = {expected_y}, got {y.value}"
    assert abs(x.grad - expected_dy_dx) < 1e-10, f"Expected f'(π/6) = {expected_dy_dx}, got {x.grad}"
    print("✅ Trigonometric function test passed!")

def test_chain_rule_example() -> None:
    """Test the chain rule example from README: f(x) = sin(x²)."""
    print("\n=== Testing Chain Rule Example: f(x) = sin(x²) ===")
    
    x = Value(2.0, label='x')
    y = (x**2).sin()  # f(x) = sin(x²)
    y.backward()
    
    print(f"f(x) = sin(x²)")
    print(f"f({x.value}) = {y.value}")
    print(f"f'({x.value}) = {x.grad}")
    
    # Verify expected results
    expected_y = math.sin(2.0**2)  # sin(4)
    expected_dy_dx = math.cos(2.0**2) * 2 * 2.0  # cos(4) * 4
    
    assert abs(y.value - expected_y) < 1e-10, f"Expected f(2) = {expected_y}, got {y.value}"
    assert abs(x.grad - expected_dy_dx) < 1e-10, f"Expected f'(2) = {expected_dy_dx}, got {x.grad}"
    print("✅ Chain rule test passed!")

def main() -> None:
    """Run all README example tests."""
    print("Running all README example tests...")
    print("Note: This implementation uses reverse-mode autodiff (backpropagation)\n")
    
    try:
        test_basic_usage()
        test_trigonometric_functions()
        test_complex_expressions()
        test_labeling_for_visualization()
        test_example_1_linear_function()
        test_example_2_quadratic_function()
        test_example_3_trigonometric_function()
        test_chain_rule_example()
        
        print("\n🎉 All README example tests passed successfully!")
        print("All code examples in the documentation are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 