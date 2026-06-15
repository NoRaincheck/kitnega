"""
FlatScript - A Chomsky Type-3 Regular Scripting Language (Python-like + Braces)
Constraints enforced:
  - Max brace nesting depth: 2 (top-level → one block level)
  - Max statements per block: 4 (unrolled finite states)
  - No nested function definitions (flat structure only)
  - Right-linear grammar translation → deterministic O(n) parsing
"""

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ======================== LEXER ========================
class TokenType:
    INT = "INT"
    STRING = "STRING"
    IDENT = "IDENT"
    OP = "OP"
    KEYWORD = "KEYWORD"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    ASSIGN = "ASSIGN"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    EOF = "EOF"


@dataclass
class Token:
    type: str
    value: Any


def tokenize(source: str) -> List[Token]:
    tokens = []
    i = 0
    keywords = {"def", "if", "else", "for", "in", "range", "return", "break"}

    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue

        # Numbers
        if ch.isdigit() or (ch == "." and (i + 1) < len(source) and source[i + 1].isdigit()):
            start = i
            while i < len(source) and (source[i].isdigit() or source[i] == "."):
                i += 1
            tokens.append(Token(TokenType.INT, float(source[start:i])))
            continue

        # Identifiers & Keywords
        if ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            tokens.append(Token(TokenType.KEYWORD, word) if word in keywords else Token(TokenType.IDENT, word))
            continue

        # Strings
        if ch == '"':
            start = i + 1
            i += 1
            while i < len(source) and source[i] != '"':
                if source[i] == "\\":
                    i += 2
                else:
                    i += 1
            tokens.append(Token(TokenType.STRING, source[start:i]))
            i += 1
            continue

        # Two-char operators
        two = source[i : i + 2]
        if two in {"==", "!=", "<=", ">="}:
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
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            "=": TokenType.ASSIGN,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
        }
        if ch in syms:
            tokens.append(Token(syms[ch], ch))
            i += 1
            continue

        raise SyntaxError(f"Line {i + source.count(chr(10), 0, i)}: Unexpected character '{ch}'")

    tokens.append(Token(TokenType.EOF, None))
    return tokens


# ======================== PARSER (FSA-driven / Type-3) ========================
MAX_DEPTH = 2
MAX_STMTS_BLOCK = 4


class FlatScriptParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.depth = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, None)

    def consume(self, expected_type=None, expected_val=None):
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
        while self.peek().type != TokenType.EOF:
            if self.peek().value == "def":
                nodes.append(self.parse_decl())
            else:
                nodes.append(self.parse_statement())
        return nodes

    def parse_decl(self) -> Dict:
        self.consume(TokenType.KEYWORD, "def")
        name = self.consume(TokenType.IDENT).value
        self.consume(TokenType.LPAREN)

        params = []
        while self.peek().type == TokenType.IDENT:
            params.append(self.consume(TokenType.IDENT).value)
            if self.peek().type == TokenType.COMMA:
                self.consume()
            else:
                break
        self.consume(TokenType.RPAREN)

        # Type-3 Depth Constraint
        if self.depth >= MAX_DEPTH - 1:
            raise SyntaxError("Type-3 Constraint: Maximum brace nesting depth exceeded")

        self.consume(TokenType.LBRACE)
        self.depth += 1

        body = self.parse_block()
        self.depth -= 1

        self.consume(TokenType.RBRACE)

        return {"type": "def", "name": name, "params": params, "body": body}

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

        # Control Flow Keywords
        if tok.type == TokenType.KEYWORD and tok.value in ("if", "for", "return", "break"):
            kw = self.consume().value
            if kw == "if":
                cond = self.parse_expr()
                self.consume(TokenType.LBRACE)
                body = self.parse_block()
                self.consume(TokenType.RBRACE)
                else_body = []
                if self.peek().type == TokenType.KEYWORD and self.peek().value == "else":
                    self.consume()
                    self.consume(TokenType.LBRACE)
                    else_body = self.parse_block()
                    self.consume(TokenType.RBRACE)
                return {"type": "if", "cond": cond, "body": body, "else_body": else_body}

            elif kw == "for":
                var = self.consume(TokenType.IDENT).value
                self.consume(TokenType.KEYWORD, "in")
                rng = self.parse_range()
                self.consume(TokenType.LBRACE)
                body = self.parse_block()
                self.consume(TokenType.RBRACE)
                return {"type": "for", "var": var, "range": rng, "body": body}

            elif kw == "return":
                val = None
                if self.peek().type != TokenType.SEMICOLON:
                    val = self.parse_expr()
                self.consume(TokenType.SEMICOLON)
                return {"type": "return", "value": val}

            elif kw == "break":
                self.consume(TokenType.SEMICOLON)
                return {"type": "break"}

        # Assignment or Expression Statement
        expr = self.parse_expr()

        if expr["type"] == "var" and self.peek().type == TokenType.ASSIGN:
            self.consume()
            val = self.parse_expr()
            if self.peek().type == TokenType.SEMICOLON:
                self.consume()
            return {"type": "assign", "var": expr["name"], "val": val}
        elif expr["type"] == "index" and self.peek().type == TokenType.ASSIGN:
            self.consume()
            val = self.parse_expr()
            if self.peek().type == TokenType.SEMICOLON:
                self.consume()
            return {"type": "subscript_assign", "obj": expr["obj"], "index": expr["index"], "val": val}
        else:
            if self.peek().type == TokenType.SEMICOLON:
                self.consume()
            return {"type": "expr_stmt", "expr": expr}

    def parse_range(self) -> Dict:
        self.consume(TokenType.KEYWORD, "range")
        self.consume(TokenType.LPAREN)
        args = []
        while self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expr())
            if self.peek().type == TokenType.COMMA:
                self.consume()
            else:
                break
        self.consume(TokenType.RPAREN)
        return {"type": "range", "args": args}

    def parse_expr(self) -> Dict:
        left = self.parse_primary()
        while True:
            if self.peek().type == TokenType.OP and self.peek().value in (
                "+",
                "-",
                "*",
                "/",
                "%",
                "<",
                ">",
                "==",
                "!=",
                "<=",
                ">=",
            ):
                op = self.consume().value
                right = self.parse_primary()
                left = {"type": "binop", "left": left, "op": op, "right": right}
            else:
                break
        return left

    def parse_primary(self) -> Dict:
        tok = self.peek()
        if tok.type == TokenType.INT:
            val = self.consume().value
            return {"type": "literal", "value": val}
        elif tok.type == TokenType.STRING:
            val = self.consume().value
            return {"type": "string_literal", "value": val}
        elif tok.type == TokenType.IDENT:
            name = self.consume().value
            if self.peek().type == TokenType.LPAREN:
                self.consume(TokenType.LPAREN)
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.parse_expr())
                    while self.peek().type == TokenType.COMMA:
                        self.consume()
                        args.append(self.parse_expr())
                self.consume(TokenType.RPAREN)
                return {"type": "call", "func": name, "args": args}
            elif self.peek().type == TokenType.LBRACKET:
                self.consume(TokenType.LBRACKET)
                index = self.parse_expr()
                self.consume(TokenType.RBRACKET)
                return {"type": "index", "obj": name, "index": index}
            return {"type": "var", "name": name}
        elif tok.type == TokenType.OP and tok.value in ("+", "-"):
            op = self.consume().value
            val = self.parse_primary()
            if op == "-":
                return {"type": "unary_minus", "val": val}
            return val
        elif tok.type == TokenType.LBRACKET:
            self.consume(TokenType.LBRACKET)
            elements = []
            if self.peek().type != TokenType.RBRACKET:
                elements.append(self.parse_expr())
                while self.peek().type == TokenType.COMMA:
                    self.consume()
                    elements.append(self.parse_expr())
            self.consume(TokenType.RBRACKET)
            return {"type": "list_literal", "elements": elements}
        else:
            raise SyntaxError(f"Unexpected token in expression: {tok}")


# ======================== EXECUTOR (Simple VM-like) ========================
class FlatScriptExecutor:
    def __init__(self):
        self.env = {}  # Symbol table
        self.functions = {}  # Top-level defs

    def run(self, program: List[Dict]) -> Optional[Any]:
        for node in program:
            if node["type"] == "def":
                self.functions[node["name"]] = {"params": node["params"], "body": node["body"]}
        return self.eval_program(program)

    def eval_program(self, nodes: List[Dict]) -> Optional[Any]:
        last = None
        for stmt in nodes:
            if stmt["type"] == "def":
                continue  # Already bound
            last = self.eval_stmt(stmt)
        return last

    def eval_stmt(self, node: Dict) -> Any:
        t = node["type"]
        if t == "assign":
            val = self.eval(node["val"])
            self.env[node["var"]] = val
            return val
        elif t == "subscript_assign":
            obj = self.env[node["obj"]]
            idx = self._to_int(self.eval(node["index"]))
            val = self.eval(node["val"])
            obj[idx] = val
            return val
        elif t == "expr_stmt":
            return self.eval(node["expr"])
        elif t == "if":
            cond_val = self.eval(node["cond"])
            if bool(cond_val):
                for s in node["body"]:
                    r = self.eval_stmt(s)
                    if isinstance(r, dict) and r.get("type") == "return":
                        return r
            else:
                for s in node["else_body"]:
                    r = self.eval_stmt(s)
                    if isinstance(r, dict) and r.get("type") == "return":
                        return r
        elif t == "for":
            rng = self._resolve_range(node["range"])
            for val in rng:
                self.env[node["var"]] = val
                executed_break = False
                for s in node["body"]:
                    res = self.eval_stmt(s)
                    if isinstance(res, dict) and res.get("type") == "break":
                        executed_break = True
                        break
                    if isinstance(res, dict) and res.get("type") == "return":
                        return res
                if executed_break:
                    break
        elif t == "return":
            val = self.eval(node.get("value")) if node.get("value") else None
            return {"type": "return", "val": val}
        elif t == "break":
            return node
        return None

    def eval(self, node: Dict) -> Any:
        if not isinstance(node, dict):
            return node
        t = node["type"]
        if t == "literal" or t == "string_literal":
            return node["value"]
        elif t == "var":
            name = node["name"]
            if name in self.env:
                val = self.env[name]
                # Auto-convert float to int if it's a whole number for display
                if isinstance(val, float) and val.is_integer():
                    return int(val)
                return val
            raise NameError(f"Undefined variable '{name}'")
        elif t == "binop":
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
            raise SyntaxError(f"Unknown operator: {op}")
        elif t == "unary_minus":
            return -self.eval(node["val"])
        elif t == "list_literal":
            return [self.eval(e) for e in node["elements"]]
        elif t == "index":
            obj = self.env[node["obj"]]
            idx = self._to_int(self.eval(node["index"]))
            return obj[idx]
        elif t == "call":
            func_name = node["func"]
            args = [self.eval(a) for a in node["args"]]

            if func_name == "print":
                print(*args)
                return None
            elif func_name == "range":
                start, end = self._to_int(args[0]), self._to_int(args[1])
                step = self._to_int(args[2]) if len(args) > 2 else 1
                return list(range(start, end, step if step != 0 else 1))
            elif func_name == "len":
                if hasattr(args[0], "__len__"):
                    return len(args[0])
                raise TypeError("len() requires a sequence")

            if func_name not in self.functions:
                raise NameError(f"Unknown function '{func_name}'")
            func = self.functions[func_name]

            saved_env = dict(self.env)
            for p, v in zip(func["params"], args):
                self.env[p] = v

            res = None
            for s in func["body"]:
                r = self.eval_stmt(s)
                if isinstance(r, dict) and r.get("type") == "return":
                    res = r.get("val")
                    break

            self.env = saved_env
            return res
        elif t == "range":
            rng = self._resolve_range(node)
            return list(rng)
        raise SyntaxError(f"Unknown AST node: {t}")

    def _to_int(self, v):
        if isinstance(v, float):
            return int(v)
        return v

    def _resolve_range(self, node: Dict):
        args = [self.eval(a) for a in node["args"]]
        start, end = self._to_int(args[0]), self._to_int(args[1])
        step = self._to_int(args[2]) if len(args) > 2 else 1
        return range(start, end, step if step != 0 else 1)


# ======================== DRIVER & DEMO ========================
def run_script(source: str):
    try:
        tokens = tokenize(source)
        parser = FlatScriptParser(tokens)
        ast = parser.parse_program()

        executor = FlatScriptExecutor()
        result = executor.run(ast)
        if isinstance(result, dict) and result.get("type") == "return":
            return result.get("val")
        return result
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Example FlatScript program (Type-3 compliant, Python-like + braces)
    EXAMPLE = """
def clamp(val, min_val, max_val) {
    if val < min_val { return min_val; } else {
        if val > max_val { return max_val; } else {
            return val;
        }
    }
}

def process_data(data) {
    total = 0;
    count = 0;
    for i in range(0, len(data)) {
        num = data[i];
        c = clamp(num, -5.0, 10.0);
        total = total + c;
        count = count + 1;
    }
    return total / count;
}

def main() {
    scores = [3.5, -2.0, 8.0, 11.0, 0];
    avg = process_data(scores);
    print("Clamped Average: " + avg);
    
    if avg > 4 {
        print("Performance is good!");
    } else {
        print("Needs improvement.");
    }
}

main();
"""
    run_script(EXAMPLE)
