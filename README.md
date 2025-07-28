# Autodiff Engine

A Python implementation of automatic differentiation (autodiff) engine that supports reverse-mode differentiation (backpropagation) with computational graph visualization.

## Features

- **Reverse-Mode Automatic Differentiation**: Efficiently computes gradients by propagating derivatives backward through the computational graph (backpropagation)
- **Computational Graph Visualization**: Generates SVG visualizations of the computational graphs
- **Mathematical Operations**: Supports arithmetic, trigonometric, exponential, and logarithmic operations
- **Chain Rule Support**: Automatically handles complex nested expressions using the chain rule
- **Type Annotations**: Full type hints for better code maintainability and IDE support
- **Labeled Values**: Value objects can be labeled for better graph visualization and debugging

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd autodiff_engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from autodiff_engine.value import Value

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
```

### Trigonometric Functions

```python
from autodiff_engine.value import Value
import math

x = Value(math.pi / 4, label='x')
y = x.sin() + x.cos()
y.backward()

print(f"y = {y.value}")
print(f"∂y/∂x = {x.grad}")
```

### Complex Expressions

```python
from autodiff_engine.value import Value

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
```

## Project Structure

```
autodiff_engine/
├── value.py              # Core Value class with autodiff implementation
├── operations.py         # Mathematical operations and their derivatives
├── test_autodiff.py     # Basic autodiff tests
├── test_trigonometric.py # Trigonometric function tests
├── test_readme_examples.py # Tests for all README examples
├── example_with_labels.py # Example demonstrating labeling functionality
├── requirements.txt      # Python dependencies
└── README.md           # This file
```

## Core Components

### Value Class

The `Value` class is the core of the autodiff engine:

- **value**: The actual numerical value (float)
- **grad**: The gradient with respect to this value (float, initialized to 0.0)
- **op**: The operation that created this value (Operation object or None for leaf nodes)
- **prev**: References to parent nodes in the computational graph (Tuple[Value, ...])
- **label**: Optional human-readable label for better graph visualization
- **_backward**: Internal callback function for custom backward logic

### Operation Class

Each mathematical operation is implemented as a class inheriting from the abstract `Operation` base class:

- **forward()**: Computes the function value
- **backward()**: Computes the gradients with respect to inputs
- **__str__()**: String representation for graph visualization

### Supported Operations

- **Arithmetic**: `+`, `-`, `*`, `/`, `**`, `neg`
- **Trigonometric**: `sin`, `cos`, `tan`, `cot`, `sinh`, `cosh`, `tanh`, `coth`
- **Exponential/Logarithmic**: `exp`, `log`

### Computational Graph

The engine automatically builds a computational graph as you perform operations. Each `Value` object maintains references to its parent nodes (`prev`) and the operation that created it (`op`), enabling efficient gradient computation through backpropagation.

### Labeling for Visualization

You can optionally label `Value` objects to make the computational graph visualization more readable:

```python
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
result.visualize('labeled_graph')
```

When labels are provided, they will be displayed in the graph nodes instead of the raw values, making it easier to understand the computation flow. Input nodes (leaf nodes) can be labeled at creation time, while intermediate nodes need to be labeled manually after creation.

## Testing

Run the test suite:

```bash
python test_autodiff.py
python test_trigonometric.py
python test_readme_examples.py
```

The `test_readme_examples.py` file contains all the code examples from this README to ensure they work correctly.

## Examples

### Example 1: Simple Linear Function

```python
from autodiff_engine.value import Value

x = Value(2.0, label='x')
y = 3 * x + 1
y.backward()

print(f"f(x) = 3x + 1")
print(f"f({x.value}) = {y.value}")
print(f"f'({x.value}) = {x.grad}")  # Should be 3
```

### Example 2: Quadratic Function

```python
from autodiff_engine.value import Value

x = Value(3.0, label='x')
y = x**2 + 2*x + 1
y.backward()

print(f"f(x) = x² + 2x + 1")
print(f"f({x.value}) = {y.value}")
print(f"f'({x.value}) = {x.grad}")  # Should be 2x + 2 = 8
```

### Example 3: Trigonometric Function

```python
from autodiff_engine.value import Value
import math

x = Value(math.pi / 6, label='x')  # 30 degrees
y = x.sin()
y.backward()

print(f"f(x) = sin(x)")
print(f"f({x.value}) = {y.value}")
print(f"f'({x.value}) = {x.grad}")  # Should be cos(x)
```

## Mathematical Background

### Reverse-Mode Automatic Differentiation (Backpropagation)

This implementation uses reverse-mode autodiff, which computes gradients by:
1. **Forward Pass**: Computing function values and building the computational graph
2. **Backward Pass**: Propagating gradients backward through the graph using the chain rule

Reverse-mode autodiff is particularly efficient for functions with many inputs and few outputs, making it ideal for machine learning applications.

### Chain Rule

The chain rule is fundamental to autodiff:
```
∂f/∂x = ∂f/∂u * ∂u/∂x
```

Where `u` is an intermediate variable in the computational graph.

### Example Chain Rule Application

For the function `f(x) = sin(x²)`:
1. Let `u = x²`, then `f(x) = sin(u)`
2. `∂f/∂u = cos(u) = cos(x²)`
3. `∂u/∂x = 2x`
4. `∂f/∂x = cos(x²) * 2x`

## Performance Considerations

- **Memory Usage**: Each `Value` object stores the computational graph, which can grow large for complex expressions
- **Computation**: Reverse-mode autodiff is efficient for functions with many inputs and few outputs (typical in machine learning)
- **Scalability**: For large-scale applications, consider using established libraries like PyTorch or TensorFlow

## Acknowledgments

This implementation is inspired by the micrograd project and serves as an educational tool for understanding automatic differentiation principles. 