"""Regression tests for conservative JSON LaTeX recovery."""
import json
import pytest
from pdf_transcriber_gemini.assembler import load_gemini_json
from pdf_transcriber_gemini.latex_safety import recover_json_latex, latex_warnings


def wrap_math(expr):
    return '{"markdown":"# Paper\\n$$' + expr + '$$","figures":[],"problems":[]}'


def test_repair_escaped_commands_and_keep_original(tmp_path, capsys):
    raw = wrap_math(r'\text{a}+\beta+\frac{1}{2}+\nu+\rho+\sim')
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    file = tmp_path / "response.json"
    file.write_text(raw, encoding="utf-8")
    data = load_gemini_json(file)
    assert data["markdown"].endswith(
        r'$$\text{a}+\beta+\frac{1}{2}+\nu+\rho+\sim$$')
    assert file.read_text(encoding="utf-8") == raw
    assert "Recovered 6" in capsys.readouterr().err


def test_good_json_is_bitwise_identical():
    raw = json.dumps({"markdown": r'$$\text{a}+\frac{1}{2}$$', "figures": [], "problems": []})
    assert recover_json_latex(raw) == (raw, [])


def test_outside_math_and_metadata_never_repaired():
    raw = r'{"markdown":"outside \text; math $\beta$","problems":["\text"],"figures":[]}'
    fixed, repairs = recover_json_latex(raw)
    assert [r.command for r in repairs] == [r'\beta']
    assert r'outside \text' in fixed
    assert r'"problems":["\text"]' in fixed


def test_unknown_command_is_not_guessed(tmp_path):
    raw = wrap_math(r'\unrecognised{a}')
    assert recover_json_latex(raw) == (raw, [])
    p = tmp_path / "bad.json"
    p.write_text(raw)
    with pytest.raises(ValueError, match="invalid JSON"):
        load_gemini_json(p)


def test_legitimate_latex_linebreak_preserved():
    raw = json.dumps({"markdown": r'$$\begin{aligned} a&=1\\b&=2 \end{aligned}$$',
                      "figures": [], "problems": []})
    assert recover_json_latex(raw) == (raw, [])


def test_warnings_do_not_change_math(tmp_path, capsys):
    bad = "$" + r'\\text{a} + x\_y' + "$"
    raw = json.dumps({"markdown": bad, "figures": [], "problems": []})
    p = tmp_path / "warn.json"
    p.write_text(raw)
    assert load_gemini_json(p)["markdown"] == bad
    output = capsys.readouterr().err
    assert "double backslash" in output
    assert "escaped underscore" in output


def test_control_chars_and_unbalanced_braces_detected():
    issues = latex_warnings("$x\t imes y$ and $\\frac{1{2}$" + "\b\f")
    assert any("tab inside math" in w.message for w in issues)
    assert any("brace" in w.message for w in issues)
    assert any("U+0008" in w.message for w in issues)


def test_raw_dollar_escape_not_a_math_delimiter():
    raw = r'{"markdown":"Price \\$5 and \text{word}; math $\beta$","figures":[],"problems":[]}'
    assert [e.command for e in recover_json_latex(raw)[1]] == [r'\beta']
