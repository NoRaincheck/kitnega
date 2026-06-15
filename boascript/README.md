# boascript

Two Chomsky Type-3 Regular Scripting Languages with different syntax families:
**FlatScript** (Python-like) and **AglScript** (Go).

Both enforce finite-state grammar constraints while preserving usable scripting
semantics.

---

## Shared Type-3 Constraints

| Constraint       | FlatScript   | AglScript    |
| ---------------- | ------------ | ------------ |
| Max brace depth  | 2            | 3            |
| Max stmts/block  | 4            | 4            |
| Nested funcs     | No           | No           |
| Grammar type     | Right-linear | Right-linear |
| Parse complexity | O(n) DFA     | O(n) DFA     |

---

## Syntax Differences

| Feature       | FlatScript             | AglScript                             |
| ------------- | ---------------------- | ------------------------------------- |
| Entry point   | `main()` call          | `func main()`                         |
| Package decl  | None                   | `package main` required               |
| Func decl     | `def name(args) {`     | `func name(args) type {`              |
| Parameters    | `a, b`                 | `a int, b string`                     |
| Return type   | Implicit               | Explicit (`int`, `string`, `float64`) |
| Variable decl | `x = 10`               | `x := 10` or `var x int = 10`         |
| For loop      | `for i in range(0, n)` | `for i := 0; i < n; i++`              |
| Semicolons    | Required               | Optional (newline insertion)          |
| Comments      | None                   | `// line comment`                     |
| Slice literal | `[1, 2, 3]`            | `[]int{1, 2, 3}`                      |
| Print         | `print("msg")`         | `println("msg")`                      |
| Types         | Dynamic                | Static (int, string, float64, []T)    |

---

## AglScript Notes

- Valid Go syntax subset — programs can be transpiled to full Go
- `package main` + `func main()` required for entry
- Explicit types on params, returns, and `var` declarations
- `:=` short declaration syntax supported
- Slice literals use Go's `[]type{elements}` form
- For loops use C-style `init; cond; post` (not `for x in range`)
- `println`, `len`, `append` are builtins (no imports needed)
- Post-increment `i++` supported in for-loop post clause
- Comments (`//`) are ignored by lexer

---

## FlatScript Notes

- Python-inspired syntax with brace delimiters
- Dynamic typing — no type annotations
- `def` for functions, `range()` for iteration
- Explicit semicolons required
- Lists use `[1, 2, 3]` bracket syntax
- `print()` for output, `len()` for length
- Simpler grammar — lower nesting limit (depth 2)

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
uv run python boascript/flatscript.py
uv run python boascript/aglscript.py
```
