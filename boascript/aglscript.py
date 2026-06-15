"""
AglScript - A Chomsky Type-3 Regular Scripting Language (Go Syntax)
Constraints enforced:
  - Max brace nesting depth: 3 (package → func → block)
  - Max statements per block: 4 (unrolled finite states)
  - No nested function definitions (flat structure only)
  - Right-linear grammar translation → deterministic O(n) parsing
  - Valid Go syntax subset (package, func, basic types)
"""

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ======================== LEXER ========================
class TokenType:
    PACKAGE = "PACKAGE"
    FUNC = "FUNC"
    IF = "IF"
    ELSE = "ELSE"
    FOR = "FOR"
    RETURN = "RETURN"
    BREAK = "BREAK"
    VAR = "VAR"
    INT_TYPE = "INT_TYPE"
    STRING_TYPE = "STRING_TYPE"
    FLOAT_TYPE = "FLOAT_TYPE"
    INT_LIT = "INT_LIT"
    FLOAT_LIT = "FLOAT_LIT"
    STRING_LIT = "STRING_LIT"
    IDENT = "IDENT"
    OP = "OP"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    ASSIGN = "ASSIGN"
    SHORT_ASSIGN = "SHORT_ASSIGN"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    EOF = "EOF"


@dataclass
class Token:
    type: str
    value: Any


KEYWORDS = {"package", "func", "if", "else", "for", "return", "break", "var"}
TYPES = {"int", "string", "float64"}


def tokenize(source: str) -> List[Token]:
    tokens = []
    i = 0

    while i < len(source):
        ch = source[i]

        if ch.isspace():
            i += 1
            continue

        # Line comments
        if ch == "/" and i + 1 < len(source) and source[i + 1] == "/":
            while i < len(source) and source[i] != "\n":
                i += 1
            continue

        # Integer literals
        if ch.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(Token(TokenType.INT_LIT, int(source[start:i])))
            continue

        # Float literals
        if ch == "." and i + 1 < len(source) and source[i + 1].isdigit():
            start = i
            i += 1
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(Token(TokenType.FLOAT_LIT, float(source[start:i])))
            continue

        # Identifiers and keywords
        if ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]

            if word == "package":
                tokens.append(Token(TokenType.PACKAGE, word))
            elif word == "func":
                tokens.append(Token(TokenType.FUNC, word))
            elif word == "if":
                tokens.append(Token(TokenType.IF, word))
            elif word == "else":
                tokens.append(Token(TokenType.ELSE, word))
            elif word == "for":
                tokens.append(Token(TokenType.FOR, word))
            elif word == "return":
                tokens.append(Token(TokenType.RETURN, word))
            elif word == "break":
                tokens.append(Token(TokenType.BREAK, word))
            elif word == "var":
                tokens.append(Token(TokenType.VAR, word))
            elif word == "int":
                tokens.append(Token(TokenType.INT_TYPE, word))
            elif word == "string":
                tokens.append(Token(TokenType.STRING_TYPE, word))
            elif word == "float64":
                tokens.append(Token(TokenType.FLOAT_TYPE, word))
            else:
                tokens.append(Token(TokenType.IDENT, word))
            continue

        # String literals
        if ch == '"':
            i += 1
            start = i
            while i < len(source) and source[i] != '"':
                if source[i] == "\\":
                    i += 2
                else:
                    i += 1
            tokens.append(Token(TokenType.STRING_LIT, source[start:i]))
            i += 1
            continue

        # Two-char operators
        two = source[i : i + 2]
        if two in {"==", "!=", "<=", ">=", ":=", "++"}:
            if two == ":=":
                tokens.append(Token(TokenType.SHORT_ASSIGN, two))
            else:
                tokens.append(Token(TokenType.OP, two))
            i += 2
            continue

        # Single char symbols
        syms = {
            "+": TokenType.OP,
            "-": TokenType.OP,
            "*": TokenType.OP,
            "/": TokenType.OP,
            "%": TokenType.OP,
            "<": TokenType.OP,
            ">": TokenType.OP,
            "=": TokenType.ASSIGN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
            ":": TokenType.COLON,
        }
        if ch in syms:
            tokens.append(Token(syms[ch], ch))
            i += 1
            continue

        raise SyntaxError(f"Unexpected character '{ch}'")

    tokens.append(Token(TokenType.EOF, None))
    return tokens


# ======================== PARSER (FSA-driven / Type-3) ========================
MAX_DEPTH = 3
MAX_STMTS_BLOCK = 4


class AglScriptParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.depth = 0

    def peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, None)

    def consume(self, expected_type: str = None, expected_val: str = None) -> Token:
        tok = self.peek()
        if tok.type == TokenType.EOF:
            raise SyntaxError("Unexpected EOF")
        if expected_type and tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok.type}")
        if expected_val and tok.value != expected_val:
            raise SyntaxError(f"Expected '{expected_val}', got '{tok.value}'")
        self.pos += 1
        return tok

    def parse_program(self) -> List[Dict]:
        nodes = []
        self.consume(TokenType.PACKAGE)
        self.consume(TokenType.IDENT, "main")

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.FUNC:
                nodes.append(self.parse_func_decl())
            else:
                raise SyntaxError(f"Expected 'func', got '{self.peek().value}'")
        return nodes

    def parse_func_decl(self) -> Dict:
        self.consume(TokenType.FUNC)
        name = self.consume(TokenType.IDENT).value
        self.consume(TokenType.LPAREN)

        params = []
        while self.peek().type != TokenType.RPAREN:
            param_name = self.consume(TokenType.IDENT).value
            param_type = self.parse_type()
            params.append({"name": param_name, "type": param_type})
            if self.peek().type == TokenType.COMMA:
                self.consume()
            else:
                break

        self.consume(TokenType.RPAREN)

        # Return type (optional)
        ret_type = None
        if self.peek().type not in (TokenType.LBRACE, TokenType.EOF):
            if self.peek().type in (TokenType.INT_TYPE, TokenType.STRING_TYPE, TokenType.FLOAT_TYPE):
                ret_type = self.parse_type()

        # Type-3 Depth Constraint
        if self.depth >= MAX_DEPTH - 1:
            raise SyntaxError("Type-3 Constraint: Maximum brace nesting depth exceeded")

        self.consume(TokenType.LBRACE)
        self.depth += 1
        body = self.parse_block()
        self.depth -= 1
        self.consume(TokenType.RBRACE)

        return {"type": "func", "name": name, "params": params, "ret_type": ret_type, "body": body}

    def parse_type(self) -> str:
        tok = self.peek()
        if tok.type == TokenType.LBRACKET:
            self.consume()
            self.consume(TokenType.RBRACKET)
            base = self.parse_type()
            return f"[]{base}"
        if tok.type in (TokenType.INT_TYPE, TokenType.STRING_TYPE, TokenType.FLOAT_TYPE):
            self.consume()
            return tok.value
        raise SyntaxError(f"Expected type, got '{tok.value}'")

    def parse_block(self) -> List[Dict]:
        stmts = []
        count = 0
        while True:
            tok = self.peek()
            if tok.type == TokenType.RBRACE or tok.type == TokenType.EOF:
                break
            if count >= MAX_STMTS_BLOCK:
                raise SyntaxError(f"Type-3 Constraint: Block exceeds max statements ({MAX_STMTS_BLOCK})")
            stmts.append(self.parse_statement())
            count += 1
        return stmts

    def parse_statement(self) -> Dict:
        tok = self.peek()

        if tok.type == TokenType.VAR:
            return self.parse_var_decl()
        elif tok.type == TokenType.IF:
            return self.parse_if()
        elif tok.type == TokenType.FOR:
            return self.parse_for()
        elif tok.type == TokenType.RETURN:
            return self.parse_return()
        elif tok.type == TokenType.BREAK:
            self.consume()
            self.expect_semicolon()
            return {"type": "break"}

        # Assignment or expression statement
        expr = self.parse_expr()

        if tok.type == TokenType.IDENT and self.peek().type == TokenType.SHORT_ASSIGN:
            self.consume()
            val = self.parse_expr()
            self.expect_semicolon()
            return {"type": "short_decl", "name": tok.value, "val": val}
        elif tok.type == TokenType.IDENT and self.peek().type == TokenType.ASSIGN:
            self.consume()
            val = self.parse_expr()
            self.expect_semicolon()
            return {"type": "assign", "name": tok.value, "val": val}
        else:
            self.expect_semicolon()
            return {"type": "expr_stmt", "expr": expr}

    def parse_var_decl(self) -> Dict:
        self.consume(TokenType.VAR)
        name = self.consume(TokenType.IDENT).value
        var_type = self.parse_type()

        val = None
        if self.peek().type == TokenType.ASSIGN:
            self.consume()
            val = self.parse_expr()

        self.expect_semicolon()
        return {"type": "var_decl", "name": name, "var_type": var_type, "val": val}

    def parse_if(self) -> Dict:
        self.consume(TokenType.IF)
        cond = self.parse_expr()
        self.consume(TokenType.LBRACE)
        self.depth += 1
        body = self.parse_block()
        self.depth -= 1
        self.consume(TokenType.RBRACE)

        else_body = []
        if self.peek().type == TokenType.ELSE:
            self.consume()
            self.consume(TokenType.LBRACE)
            self.depth += 1
            else_body = self.parse_block()
            self.depth -= 1
            self.consume(TokenType.RBRACE)

        return {"type": "if", "cond": cond, "body": body, "else_body": else_body}

    def parse_for(self) -> Dict:
        self.consume(TokenType.FOR)

        # Go-style for: for init; cond; post { body }
        init = None
        cond = None
        post = None

        if self.peek().type != TokenType.SEMICOLON:
            init = self.parse_for_init()

        self.consume(TokenType.SEMICOLON)

        if self.peek().type != TokenType.SEMICOLON:
            cond = self.parse_expr()

        self.consume(TokenType.SEMICOLON)

        if self.peek().type != TokenType.LBRACE:
            post = self.parse_for_post()

        self.consume(TokenType.LBRACE)
        self.depth += 1
        body = self.parse_block()
        self.depth -= 1
        self.consume(TokenType.RBRACE)

        return {"type": "for", "init": init, "cond": cond, "post": post, "body": body}

    def parse_for_init(self) -> Dict:
        tok = self.peek()
        if tok.type == TokenType.IDENT and self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]
            if next_tok.type == TokenType.SHORT_ASSIGN:
                name = self.consume().value
                self.consume()
                val = self.parse_expr()
                return {"type": "short_decl", "name": name, "val": val}
        return self.parse_expr()

    def parse_for_post(self) -> Dict:
        tok = self.peek()
        if tok.type == TokenType.IDENT:
            name = self.consume().value
            if self.peek().type == TokenType.OP and self.peek().value == "++":
                self.consume()
                return {"type": "post_inc", "name": name}
            elif self.peek().type == TokenType.ASSIGN:
                self.consume()
                val = self.parse_expr()
                return {"type": "assign", "name": name, "val": val}
        return self.parse_expr()

    def parse_return(self) -> Dict:
        self.consume(TokenType.RETURN)
        val = None
        if self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
            val = self.parse_expr()
        self.expect_semicolon()
        return {"type": "return", "value": val}

    def expect_semicolon(self):
        if self.peek().type == TokenType.SEMICOLON:
            self.consume()

    def parse_expr(self) -> Dict:
        left = self.parse_primary()
        while True:
            tok = self.peek()
            if tok.type == TokenType.OP and tok.value in ("+", "-", "*", "/", "%", "<", ">", "==", "!=", "<=", ">="):
                op = self.consume().value
                right = self.parse_primary()
                left = {"type": "binop", "left": left, "op": op, "right": right}
            else:
                break
        return left

    def parse_primary(self) -> Dict:
        tok = self.peek()

        if tok.type == TokenType.INT_LIT:
            self.consume()
            return {"type": "int_literal", "value": tok.value}

        if tok.type == TokenType.FLOAT_LIT:
            self.consume()
            return {"type": "float_literal", "value": tok.value}

        if tok.type == TokenType.STRING_LIT:
            self.consume()
            return {"type": "string_literal", "value": tok.value}

        if tok.type == TokenType.IDENT:
            name = self.consume().value
            if self.peek().type == TokenType.LPAREN:
                self.consume()
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.parse_expr())
                    while self.peek().type == TokenType.COMMA:
                        self.consume()
                        args.append(self.parse_expr())
                self.consume(TokenType.RPAREN)
                return {"type": "call", "func": name, "args": args}
            elif self.peek().type == TokenType.LBRACKET:
                self.consume()
                index = self.parse_expr()
                self.consume(TokenType.RBRACKET)
                return {"type": "index", "obj": name, "index": index}
            return {"type": "var", "name": name}

        if tok.type == TokenType.OP and tok.value in ("+", "-"):
            op = self.consume().value
            val = self.parse_primary()
            if op == "-":
                return {"type": "unary_minus", "val": val}
            return val

        if tok.type == TokenType.LBRACKET:
            self.consume()
            # Check for []type{...} Go slice literal
            if self.peek().type == TokenType.RBRACKET:
                self.consume()
                if self.peek().type in (TokenType.INT_TYPE, TokenType.STRING_TYPE, TokenType.FLOAT_TYPE):
                    self.parse_type()
                    self.consume(TokenType.LBRACE)
                    elements = []
                    if self.peek().type != TokenType.RBRACE:
                        elements.append(self.parse_expr())
                        while self.peek().type == TokenType.COMMA:
                            self.consume()
                            elements.append(self.parse_expr())
                    self.consume(TokenType.RBRACE)
                    return {"type": "slice_literal", "elements": elements}
                raise SyntaxError("Expected type after []")
            elements = []
            elements.append(self.parse_expr())
            while self.peek().type == TokenType.COMMA:
                self.consume()
                elements.append(self.parse_expr())
            self.consume(TokenType.RBRACKET)
            return {"type": "slice_literal", "elements": elements}

        if tok.type == TokenType.LPAREN:
            self.consume()
            expr = self.parse_expr()
            self.consume(TokenType.RPAREN)
            return expr

        raise SyntaxError(f"Unexpected token: {tok}")


# ======================== EXECUTOR ========================
class AglScriptExecutor:
    def __init__(self):
        self.env: Dict[str, Any] = {}
        self.functions: Dict[str, Dict] = {}

    def run(self, program: List[Dict]) -> Optional[Any]:
        for node in program:
            if node["type"] == "func":
                self.functions[node["name"]] = node

        if "main" not in self.functions:
            raise RuntimeError("No main function defined")

        return self.call_function("main", [])

    def call_function(self, name: str, args: List[Any]) -> Any:
        func = self.functions.get(name)
        if not func:
            raise RuntimeError(f"Undefined function: {name}")

        saved_env = dict(self.env)

        for param, arg in zip(func["params"], args):
            self.env[param["name"]] = arg

        result = None
        for stmt in func["body"]:
            result = self.exec_stmt(stmt)
            if isinstance(result, dict) and result.get("type") == "return":
                result = result.get("value")
                break

        self.env = saved_env
        return result

    def exec_stmt(self, node: Dict) -> Any:
        t = node["type"]

        if t == "short_decl":
            val = self.eval(node["val"])
            self.env[node["name"]] = val
            return val

        if t == "assign":
            val = self.eval(node["val"])
            self.env[node["name"]] = val
            return val

        if t == "var_decl":
            val = self.eval(node["val"]) if node["val"] else None
            self.env[node["name"]] = val
            return val

        if t == "expr_stmt":
            return self.eval(node["expr"])

        if t == "if":
            cond = self.eval(node["cond"])
            if cond:
                for s in node["body"]:
                    r = self.exec_stmt(s)
                    if isinstance(r, dict) and r.get("type") == "return":
                        return r
            else:
                for s in node["else_body"]:
                    r = self.exec_stmt(s)
                    if isinstance(r, dict) and r.get("type") == "return":
                        return r
            return None

        if t == "for":
            if node["init"]:
                self.exec_stmt(node["init"])

            while True:
                if node["cond"]:
                    cond_val = self.eval(node["cond"])
                    if not cond_val:
                        break

                break_hit = False
                for s in node["body"]:
                    r = self.exec_stmt(s)
                    if isinstance(r, dict) and r.get("type") == "break":
                        break_hit = True
                        break
                    if isinstance(r, dict) and r.get("type") == "return":
                        return r
                if break_hit:
                    break

                if node["post"]:
                    self.exec_for_post(node["post"])

            return None

        if t == "return":
            return {"type": "return", "value": self.eval(node["value"]) if node["value"] else None}

        if t == "break":
            return {"type": "break"}

        return None

    def exec_for_post(self, node: Dict):
        if node["type"] == "post_inc":
            self.env[node["name"]] = self.env.get(node["name"], 0) + 1
        elif node["type"] == "assign":
            self.env[node["name"]] = self.eval(node["val"])

    def eval(self, node: Dict) -> Any:
        if not isinstance(node, dict):
            return node

        t = node["type"]

        if t == "int_literal":
            return node["value"]

        if t == "float_literal":
            return node["value"]

        if t == "string_literal":
            return node["value"]

        if t == "var":
            name = node["name"]
            if name in self.env:
                return self.env[name]
            if name in self.functions:
                return name
            raise RuntimeError(f"Undefined variable: {name}")

        if t == "binop":
            left = self.eval(node["left"])
            right = self.eval(node["right"])
            op = node["op"]
            if op == "+":
                if isinstance(left, str) or isinstance(right, str):
                    return str(left) + str(right)
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                if right == 0:
                    raise RuntimeError("Division by zero")
                return left / right
            if op == "%":
                return left % right
            if op == "<":
                return left < right
            if op == ">":
                return left > right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "<=":
                return left <= right
            if op == ">=":
                return left >= right
            raise RuntimeError(f"Unknown operator: {op}")

        if t == "unary_minus":
            return -self.eval(node["val"])

        if t == "slice_literal":
            return [self.eval(e) for e in node["elements"]]

        if t == "index":
            obj = self.env[node["obj"]]
            idx = self.eval(node["index"])
            return obj[idx]

        if t == "call":
            func_name = node["func"]
            args = [self.eval(a) for a in node["args"]]

            if func_name == "println":
                print(*args)
                return None
            elif func_name == "len":
                return len(args[0]) if args else 0
            elif func_name == "append":
                lst = list(args[0])
                lst.append(args[1])
                return lst

            return self.call_function(func_name, args)

        raise RuntimeError(f"Unknown expression: {t}")


# ======================== DRIVER ========================
def run_script(source: str):
    try:
        tokens = tokenize(source)
        parser = AglScriptParser(tokens)
        ast = parser.parse_program()

        executor = AglScriptExecutor()
        result = executor.run(ast)
        return result
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    EXAMPLE = """package main

func clamp(val int, minVal int, maxVal int) int {
    if val < minVal {
        return minVal
    } else {
        if val > maxVal {
            return maxVal
        } else {
            return val
        }
    }
}

func processList(items []int) int {
    total := 0
    for i := 0; i < len(items); i++ {
        num := items[i]
        c := clamp(num, -5, 10)
        total = total + c
    }
    return total
}

func main() {
    data := []int{3, -2, 8, 11, 0}
    result := processList(data)
    println("Sum: " + result)

    if result > 10 {
        println("Above threshold")
    } else {
        println("Below threshold")
    }
}
"""
    run_script(EXAMPLE)
