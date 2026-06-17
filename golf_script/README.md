# boascript

A Chomsky Type-3 Regular Scripting Language with Go syntax.

GolfScript is designed as a minimal Go subset for learning and exercise —
similar in spirit to [monty](https://github.com/pydantic/monty/). It provides a
clean Python API for embedding Go-like code in Python applications.

---

## Type-3 Constraints

| Constraint       | GolfScript   |
| ---------------- | ------------ |
| Max brace depth  | 3            |
| Max stmts/block  | 8            |
| Nested funcs     | No           |
| Grammar type     | Right-linear |
| Parse complexity | O(n) DFA     |

---

## Features

| Feature       | GolfScript                            |
| ------------- | ------------------------------------- |
| Entry point   | `func main()`                         |
| Package decl  | `package main` required               |
| Func decl     | `func name(args) type {`              |
| Parameters    | `a int, b string`                     |
| Return type   | Explicit (`int`, `string`, `float64`) |
| Variable decl | `x := 10` or `var x int = 10`         |
| For loop      | `for i := 0; i < n; i++`              |
| Comments      | `// line comment`                     |
| Slice literal | `[]int{1, 2, 3}`                      |
| Print         | `println("msg")`                      |
| Types         | Static (int, string, float64, []T)    |

---

## Design Goals

- **Minimal Go subset**: Programs should be valid Go syntax where possible
- **Learning tool**: Focus on core language constructs, not full Go spec
- **Easy to parse**: Right-linear grammar enables deterministic O(n) parsing
- **Python API**: Clean integration with Python via the `Golf` class
- **Go conventions**: Support for idioms like `:=`, `for` loops, slice literals

---

## Language Features

### Core Language

- `package main` + `func main()` required for entry
- Explicit types on params, returns, and `var` declarations
- `:=` short declaration syntax
- Slice literals use Go's `[]type{elements}` form
- For loops use C-style `init; cond; post`
- `println`, `len`, `append` are builtins (no imports needed)
- Post-increment `i++` supported in for-loop post clause
- Comments (`//`) are ignored by lexer

### Supported Types

- `int` — integers
- `float64` — floating point
- `string` — string literals
- `[]T` — slice of type T

### Built-in Functions

- `println(args...)` — print to stdout
- `len(s)` — get length of string/slice
- `append(slice, elem)` — append element to slice

---

## Package & Module Support

### Basic Import System

```go
import "fmt"
import "strings"
```

### Module Resolution

- Built-in modules: `fmt`, `strings`, `math`
- Module registry maps import paths to implementations
- Simple path resolution without full GOPATH semantics

### Standard Library Modules (Planned)

- `fmt` — formatting functions (`Sprintf`, `Errorf`)
- `strings` — string manipulation (`Contains`, `Split`, `Join`)
- `math` — mathematical functions (`Abs`, `Max`, `Min`)

---

## Go Language Convention Support

### Error Handling (Future)

```go
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}
```

### Struct Types (Future)

```go
type Point struct {
    X int
    Y int
}
```

### Method Receivers (Future)

```go
func (p Point) String() string {
    return fmt.Sprintf("(%d, %d)", p.X, p.Y)
}
```

---

## Python Integration

The `Golf` class provides a clean Python API for embedding GolfScript in Python
applications:

### Basic Usage

```python
from boascript import Golf

src = """
package main

func add(a int, b int) int {
    return a + b
}

func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}
"""

g = Golf(src)
print(g.list_functions())  # ['add', 'factorial']

result = g.execute("add", 3, 4)
print(result)  # 7

result = g.execute("factorial", 5)
print(result)  # 120
```

### Working with Slices

```python
src = """
package main

func sum(items []int) int {
    total := 0
    for i := 0; i < len(items); i++ {
        total = total + items[i]
    }
    return total
}

func main() {}
"""

g = Golf(src)
result = g.execute("sum", [1, 2, 3, 4, 5])
print(result)  # 15
```

### Method Reference

| Method                   | Description                           |
| ------------------------ | ------------------------------------- |
| `Golf(source)`           | Create a GolfScript program           |
| `g.execute(func, *args)` | Call a function with Python arguments |
| `g.list_functions()`     | List all defined function names       |
| `g.get_function(name)`   | Get AST node for a function           |

---

## Why Type-3?

Chomsky Type-3 (regular languages) are recognized by finite automata with no
unbounded memory. Both languages achieve this by:

1. Bounding nesting depth at parse time
2. Limiting statements per block to finite states
3. Using right-linear grammar rules (no recursion in productions)
4. Runtime iteration via counters, not syntactic recursion

This guarantees deterministic O(n) parsing with zero backtracking.

---

## Running

```bash
uv run python boascript/golf_script.py
```
