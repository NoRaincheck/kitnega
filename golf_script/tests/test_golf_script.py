"""Tests for GolfScript implementation."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from golf_script import (
    Executor,
    Golf,
    Parser,
    TokenType,
    tokenize,
)


class TestLexer:
    def test_tokenize_keywords(self):
        tokens = tokenize("package func if else for return break var import")
        types = [t.type for t in tokens[:-1]]  # Exclude EOF
        assert types == [
            TokenType.PACKAGE,
            TokenType.FUNC,
            TokenType.IF,
            TokenType.ELSE,
            TokenType.FOR,
            TokenType.RETURN,
            TokenType.BREAK,
            TokenType.VAR,
            TokenType.IMPORT,
        ]

    def test_tokenize_types(self):
        tokens = tokenize("int string float64")
        types = [t.type for t in tokens[:-1]]
        assert types == [
            TokenType.INT_TYPE,
            TokenType.STRING_TYPE,
            TokenType.FLOAT_TYPE,
        ]

    def test_tokenize_literals(self):
        tokens = tokenize('42 3.14 "hello"')
        assert tokens[0].type == TokenType.INT_LIT
        assert tokens[0].value == 42
        assert tokens[1].type == TokenType.FLOAT_LIT
        assert tokens[1].value == 3.14
        assert tokens[2].type == TokenType.STRING_LIT
        assert tokens[2].value == "hello"

    def test_tokenize_operators(self):
        tokens = tokenize("+ - * / % < > == != <= >= :=")
        types = [t.type for t in tokens[:-1]]
        assert types[0] == TokenType.OP
        assert types[-1] == TokenType.SHORT_ASSIGN

    def test_tokenize_comments(self):
        tokens = tokenize("x // this is a comment\ny")
        assert len(tokens) == 3  # x, y, EOF


class TestParser:
    def test_parse_simple_program(self):
        source = """
        package main
        func main() {
            println("hello")
        }
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        assert len(ast) == 1
        assert ast[0]["type"] == "func"
        assert ast[0]["name"] == "main"

    def test_parse_function_with_params(self):
        source = """
        package main
        func add(a int, b int) int {
            return a + b
        }
        func main() {}
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        assert len(ast) == 2
        func = ast[0]
        assert func["name"] == "add"
        assert len(func["params"]) == 2
        assert func["ret_type"] == "int"

    def test_parse_for_loop(self):
        source = """
        package main
        func main() {
            for i := 0; i < 10; i++ {
                println(i)
            }
        }
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        for_stmt = ast[0]["body"][0]
        assert for_stmt["type"] == "for"


class TestExecutor:
    def test_execute_hello_world(self):
        source = """
        package main
        func main() {
            println("Hello, World!")
        }
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        executor = Executor()
        executor.run(ast)

    def test_execute_arithmetic(self):
        source = """
        package main
        func main() {
            x := 2 + 3 * 4
            println(x)
        }
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        executor = Executor()
        executor.run(ast)

    def test_execute_slice_operations(self):
        source = """
        package main
        func main() {
            nums := []int{1, 2, 3}
            nums = append(nums, 4)
            println(len(nums))
        }
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        executor = Executor()
        executor.run(ast)


class TestIntegration:
    def test_full_program(self):
        source = """
        package main

        func main() {
            data := []int{3, -2, 8, 11, 0}
            total := 0
            for i := 0; i < len(data); i++ {
                if data[i] > 0 {
                    total = total + data[i]
                }
            }
            println("Sum: " + total)
        }
        """
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        executor = Executor()
        executor.run(ast)


class TestGolfAPI:
    def test_golf_basic(self):
        src = """
        package main

        func add(a int, b int) int {
            return a + b
        }
        """
        g = Golf(src)
        assert g.list_functions() == ["add"]
        assert g.execute("add", 3, 4) == 7

    def test_golf_multiple_functions(self):
        src = """
        package main

        func double(x int) int {
            return x * 2
        }

        func triple(x int) int {
            return x * 3
        }
        """
        g = Golf(src)
        assert g.list_functions() == ["double", "triple"]
        assert g.execute("double", 5) == 10
        assert g.execute("triple", 5) == 15

    def test_golf_with_slices(self):
        src = """
        package main

        func sum(items []int) int {
            total := 0
            for i := 0; i < len(items); i++ {
                total = total + items[i]
            }
            return total
        }
        """
        g = Golf(src)
        assert g.execute("sum", [1, 2, 3, 4, 5]) == 15

    def test_golf_string_args(self):
        src = """
        package main

        func greet(name string) string {
            return "Hello, " + name + "!"
        }
        """
        g = Golf(src)
        assert g.execute("greet", "World") == "Hello, World!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
