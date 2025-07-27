# Autodiff Engine

A Python implementation of automatic differentiation (autodiff) engine that supports forward mode differentiation with computational graph visualization.

## Features

- **Forward Mode Automatic Differentiation**: Efficiently computes gradients by propagating derivatives forward through the computational graph
- **Computational Graph Visualization**: Generates SVG visualizations of the computational graphs
- **Basic Mathematical Operations**: Supports addition, subtraction, multiplication, division, and power operations
- **Trigonometric Functions**: Implements sin, cos, tan, sinh, cosh, and tanh with their derivatives
- **Exponential and Logarithmic Functions**: Supports exp and log operations
- **Chain Rule Support**: Automatically handles complex nested expressions using the chain rule
- **Type Annotations**: Full type hints for better code maintainability and IDE support

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

# Create variables
x = Value(2.0, label='x')
y = Value(3.0, label='y')

# Build computational graph
z = x * y + x**2

# Compute gradients
z.backward()

print(f"z = {z.data}")
print(f"∂z/∂x = {x.grad}")
print(f"∂z/∂y = {y.grad}")
```

### Trigonometric Functions

```python
from autodiff_engine.value import Value
import math

x = Value(math.pi / 4, label='x')
y = Value.sin(x) + Value.cos(x)
y.backward()

print(f"y = {y.data}")
print(f"∂y/∂x = {x.grad}")
```

### Complex Expressions

```python
from autodiff_engine.value import Value

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = Value(1.0, label='c')

# Complex expression: (a^2 + b) * sin(c)
result = (a**2 + b) * Value.sin(c)
result.backward()

print(f"Result = {result.data}")
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
├── requirements.txt      # Python dependencies
└── README.md           # This file
```

## Core Components

### Value Class

The `Value` class is the core of the autodiff engine:

- **Data**: Stores the actual numerical value
- **Grad**: Stores the gradient with respect to this value
- **Label**: Human-readable label for visualization
- **Children**: References to child nodes in the computational graph
- **Operation**: The operation that created this value

### Supported Operations

- **Arithmetic**: `+`, `-`, `*`, `/`, `**`
- **Trigonometric**: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`
- **Exponential/Logarithmic**: `exp`, `log`

### Computational Graph

The engine automatically builds a computational graph as you perform operations. Each `Value` object maintains references to its children and the operation that created it, enabling efficient gradient computation through backpropagation.

## Testing

Run the test suite:

```bash
python test_autodiff.py
python test_trigonometric.py
```

## Examples

### Example 1: Simple Linear Function

```python
from autodiff_engine.value import Value

x = Value(2.0, label='x')
y = 3 * x + 1
y.backward()

print(f"f(x) = 3x + 1")
print(f"f({x.data}) = {y.data}")
print(f"f'({x.data}) = {x.grad}")  # Should be 3
```

### Example 2: Quadratic Function

```python
from autodiff_engine.value import Value

x = Value(3.0, label='x')
y = x**2 + 2*x + 1
y.backward()

print(f"f(x) = x² + 2x + 1")
print(f"f({x.data}) = {y.data}")
print(f"f'({x.data}) = {x.grad}")  # Should be 2x + 2 = 8
```

### Example 3: Trigonometric Function

```python
from autodiff_engine.value import Value
import math

x = Value(math.pi / 6, label='x')  # 30 degrees
y = Value.sin(x)
y.backward()

print(f"f(x) = sin(x)")
print(f"f({x.data}) = {y.data}")
print(f"f'({x.data}) = {x.grad}")  # Should be cos(x)
```

## Mathematical Background

### Forward Mode Automatic Differentiation

Forward mode autodiff computes gradients by propagating derivatives forward through the computational graph. For each operation, we compute both the function value and its derivative simultaneously.

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
- **Computation**: Forward mode is efficient for functions with few inputs and many outputs
- **Scalability**: For large-scale applications, consider using established libraries like PyTorch or TensorFlow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

This implementation is inspired by the micrograd project and serves as an educational tool for understanding automatic differentiation principles. 