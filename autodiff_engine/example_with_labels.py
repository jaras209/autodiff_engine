#!/usr/bin/env python3
"""
Example demonstrating the labeling functionality of Value objects.
This shows how to create labeled Value objects for better graph visualization.
"""

from value import Value

def main() -> None:
    """Demonstrate Value object labeling functionality."""
    
    # Create labeled input values
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")
    
    print(f"Created labeled values: {x}, {y}")
    
    # Perform operations and manually label intermediate results
    z = x * y
    z.label = "z = x*y"  # Manually label intermediate result
    
    w = z + x
    w.label = "w = z+x"  # Manually label intermediate result
    
    result = w ** 2
    result.label = "result = w²"  # Label the final result
    
    print(f"Result: {result}")
    print(f"Result value: {result.value}")
    
    # Compute gradients
    result.backward()
    
    print(f"Gradients:")
    print(f"  dx/dresult = {x.grad}")
    print(f"  dy/dresult = {y.grad}")
    
    # Visualize the computation graph with labels
    result.visualize('graph_with_labels')
    print("Graph visualization saved as 'graph_with_labels.svg'")
    
    # Example with trigonometric functions
    print("\n--- Trigonometric Example ---")
    angle = Value(0.5, label="θ")
    sin_angle = angle.sin()
    sin_angle.label = "sin(θ)"  # Label intermediate result
    
    cos_angle = angle.cos()
    cos_angle.label = "cos(θ)"  # Label intermediate result
    
    trig_result = sin_angle + cos_angle
    trig_result.label = "sin(θ) + cos(θ)"  # Label final result
    
    print(f"sin(θ) + cos(θ) = {trig_result.value}")
    trig_result.backward()
    print(f"d/dθ(sin(θ) + cos(θ)) = {angle.grad}")
    
    # Visualize trigonometric computation
    trig_result.visualize('graph_trigonometric')
    print("Trigonometric graph saved as 'graph_trigonometric.svg'")
    
    # Example showing the difference with and without labels
    print("\n--- Comparison Example ---")
    a = Value(1.0, label="a")
    b = Value(2.0, label="b")
    
    # Without labeling intermediate results
    c1 = a + b
    d1 = c1 * a
    e1 = d1 ** 2
    e1.label = "result1"
    
    # With labeling intermediate results
    c2 = a + b
    c2.label = "c = a+b"
    d2 = c2 * a
    d2.label = "d = c*a"
    e2 = d2 ** 2
    e2.label = "result2 = d²"
    
    print("Creating comparison graphs...")
    e1.visualize('graph_no_labels')
    e2.visualize('graph_with_intermediate_labels')
    print("Comparison graphs saved as 'graph_no_labels.svg' and 'graph_with_intermediate_labels.svg'")

if __name__ == "__main__":
    main() 