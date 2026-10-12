from repomind.parsing import parse_python


def test_definitions_source_ranges_and_imports() -> None:
    raw = (
        "import os\r\nfrom .tools import helper\r\n\r\n"
        "@decorator\r\nclass A:\r\n    \"\"\"中文说明\"\"\"\r\n"
        "    @staticmethod\r\n    async def work(x: int) -> str:\r\n"
        "        def inner():\r\n            return x\r\n        return str(inner())\r\n"
        "\r\ndef same():\r\n    pass\r\ndef same():\r\n    pass\r\n"
    ).encode()
    parsed = parse_python("revision", "pkg/mod.py", "blob", raw)
    assert parsed.file.parse_status == "parsed"
    by_name = {s.qualified_name: s for s in parsed.symbols if s.name != "same"}
    assert by_name["pkg.mod.A"].docstring == "中文说明"
    assert by_name["pkg.mod.A"].start_line == 4
    assert by_name["pkg.mod.A"].definition_line == 5
    assert by_name["pkg.mod.A.work"].kind == "async_method"
    assert "x: int" in by_name["pkg.mod.A.work"].signature
    assert by_name["pkg.mod.A.work.inner"].kind == "function"
    assert by_name["pkg.mod.A.work"].source.startswith("    @staticmethod\r\n")
    same = [s for s in parsed.symbols if s.name == "same"]
    assert len(same) == 2 and same[0].id != same[1].id
    assert parsed.imports[0].names == ("os",)
    assert parsed.imports[1].module == "tools" and parsed.imports[1].level == 1
    assert parsed.chunks[0].start_line == 4


def test_nested_class_method_and_empty_file() -> None:
    raw = b"def outer():\n    class Inner:\n        def method(self):\n            pass\n"
    symbols = parse_python("rev", "mod.py", "blob", raw).symbols
    assert [s.qualified_name for s in symbols] == [
        "mod.outer", "mod.outer.Inner", "mod.outer.Inner.method"]
    assert symbols[-1].kind == "method"
    empty = parse_python("rev", "empty.py", "emptyblob", b"")
    assert empty.file.parse_status == "parsed" and not empty.symbols


def test_import_alias_is_preserved() -> None:
    parsed = parse_python("rev", "mod.py", "blob", b"import numpy as np\nfrom .x import y as z\n")
    assert parsed.imports[0].names == ("numpy as np",)
    assert parsed.imports[1].names == ("y as z",)


def test_parse_errors_are_file_diagnostics() -> None:
    syntax = parse_python("rev", "bad.py", "blob", b"def broken(:\n")
    assert syntax.file.parse_status == "syntax_error" and not syntax.symbols
    encoding = parse_python("rev", "bad_encoding.py", "blob", b"# coding: ascii\n# \xff\n")
    assert encoding.file.parse_status == "encoding_error" and not encoding.symbols
