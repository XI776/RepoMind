"""Syntactic Python indexer. It never imports or executes target code."""

import ast
from hashlib import sha256
from io import BytesIO
import tokenize

from repomind.models import CodeChunk, CodeSymbol, ImportRelation, ParsedFile, SourceFile, stable_id


def decode_python(raw: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(BytesIO(raw).readline)
    return raw.decode(encoding)


class _Collector(ast.NodeVisitor):
    def __init__(self, file_id: str, module: str, source: str) -> None:
        self.file_id = file_id
        self.module = module
        self.lines = source.splitlines(keepends=True)
        self.scope: list[str] = []
        self.class_flags: list[bool] = []
        self.symbols: list[CodeSymbol] = []
        self.chunks: list[CodeChunk] = []
        self.imports: list[ImportRelation] = []

    def _qualified_name(self, name: str) -> str:
        return ".".join((self.module, *self.scope, name))

    def _save_symbol(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        start = min((item.lineno for item in node.decorator_list), default=node.lineno)
        end = node.end_lineno or node.lineno
        source = "".join(self.lines[start - 1 : end])
        if isinstance(node, ast.ClassDef):
            args = [ast.unparse(base) for base in node.bases]
            args.extend(ast.unparse(keyword) for keyword in node.keywords)
            signature = f"class {node.name}({', '.join(args)})" if args else f"class {node.name}"
            kind = "class"
        else:
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            signature = f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"
            in_class = bool(self.class_flags and self.class_flags[-1])
            kind = ("async_method" if isinstance(node, ast.AsyncFunctionDef) else "method") if in_class else (
                "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            )
        qualified = self._qualified_name(node.name)
        symbol_id = stable_id(self.file_id, qualified, kind, str(node.lineno))
        symbol = CodeSymbol(
            id=symbol_id,
            file_id=self.file_id,
            name=node.name,
            qualified_name=qualified,
            kind=kind,
            signature=signature,
            docstring=ast.get_docstring(node),
            start_line=start,
            definition_line=node.lineno,
            end_line=end,
            source=source,
        )
        self.symbols.append(symbol)
        self.chunks.append(CodeChunk(
            id=stable_id(symbol_id, str(start), str(end)),
            symbol_id=symbol_id,
            start_line=start,
            end_line=end,
            content_hash=sha256(source.encode("utf-8")).hexdigest(),
        ))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._save_symbol(node)
        self.scope.append(node.name)
        self.class_flags.append(True)
        self.generic_visit(node)
        self.class_flags.pop()
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._save_symbol(node)
        self.scope.append(node.name)
        self.class_flags.append(False)
        self.generic_visit(node)
        self.class_flags.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Import(self, node: ast.Import) -> None:
        names = tuple(self._alias_text(alias) for alias in node.names)
        self._save_import(node, "import", None, names, 0)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = tuple(self._alias_text(alias) for alias in node.names)
        self._save_import(node, "from", node.module, names, node.level)

    @staticmethod
    def _alias_text(alias: ast.alias) -> str:
        return f"{alias.name} as {alias.asname}" if alias.asname else alias.name

    def _save_import(
        self, node: ast.Import | ast.ImportFrom, kind: str, module: str | None,
        names: tuple[str, ...], level: int,
    ) -> None:
        scope = ".".join((self.module, *self.scope))
        self.imports.append(ImportRelation(
            id=stable_id(self.file_id, kind, scope, str(node.lineno), str(node.col_offset)),
            file_id=self.file_id,
            scope=scope,
            kind=kind,
            module=module,
            names=names,
            level=level,
            line=node.lineno,
        ))


def parse_python(snapshot_id: str, path: str, blob_oid: str, raw: bytes) -> ParsedFile:
    """Return exact source spans and diagnostics for one immutable Git blob."""
    file_id = stable_id(snapshot_id, path)
    content_hash = sha256(raw).hexdigest()
    common = dict(id=file_id, snapshot_id=snapshot_id, path=path, blob_oid=blob_oid,
                  content_hash=content_hash, size_bytes=len(raw))
    try:
        source = decode_python(raw)
    except (SyntaxError, UnicodeError, LookupError) as exc:
        return ParsedFile(SourceFile(**common, parse_status="encoding_error", error=str(exc)), (), (), ())
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as exc:
        return ParsedFile(SourceFile(**common, parse_status="syntax_error", error=str(exc)), (), (), ())
    module = path[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[:-9]
    collector = _Collector(file_id, module, source)
    collector.visit(tree)
    return ParsedFile(SourceFile(**common, parse_status="parsed"),
                      tuple(collector.symbols), tuple(collector.chunks), tuple(collector.imports))
