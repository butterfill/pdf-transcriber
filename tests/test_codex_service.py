"""No-network subprocess tests for optional Codex CLI citation discovery."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from pdf_transcriber_gemini import citation_service as service
from pdf_transcriber_gemini import citations


def source(root: Path) -> Path:
    md = root / "Essay.md"
    md.write_text(
        "# Essay\n\nSmith (2020) and Smith (2020).\n\n## References\n"
        "Smith, A. (2020). *Example*.\n", encoding="utf-8",
    )
    return md


def fake_codex(root: Path) -> Path:
    """An executable stand-in for the real Codex CLI, not an API request."""
    binary = root / "codex"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,re,sys\n"
        "args=sys.argv[1:]\n"
        "pathlib.Path(os.environ['FAKE_CODEX_LOG']).write_text(json.dumps(args))\n"
        "request=sys.stdin.read()\n"
        "pathlib.Path(os.environ['FAKE_CODEX_STDIN']).write_text(request)\n"
        "if os.getenv('FAKE_CODEX_EXIT'):\n"
        " print('not authenticated; run codex login',file=sys.stderr)\n"
        " sys.exit(7)\n"
        "sid=re.search(r'SOURCE_ID: ([0-9a-f]{16})',request).group(1)\n"
        "out={'schema_version':4,'source_id':sid,'references':[{'l':6,'a':'Smith, A. (2020).','authors':['Smith, A.'],'title':'Example','year':'2020'}],'mentions':['3|Smith (2020)']}\n"
        "pathlib.Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(out))\n"
        "print('fake codex completed')\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def set_codex_env(root: Path, monkeypatch) -> None:
    fake_codex(root)
    monkeypatch.setenv("PATH", str(root) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(root / "args.json"))
    monkeypatch.setenv("FAKE_CODEX_STDIN", str(root / "stdin.txt"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_codex_uses_luna_default_and_writes_provenance(tmp_path: Path, monkeypatch):
    md = source(tmp_path)
    original = md.read_bytes()
    set_codex_env(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "_gemini_citations", lambda *a, **k: pytest.fail("Gemini called"))
    assert service.find_citations(md, codex=True) == 0
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[0] == "exec"
    assert "--ephemeral" in args
    assert "--skip-git-repo-check" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--model") + 1] == "gpt-6-luna"
    assert args[-1] == "-"
    assert "--output-schema" not in args
    prompt = (tmp_path / "stdin.txt").read_text()
    assert "OUTPUT JSON SHAPE" in prompt
    assert "LINE|TEXT" in prompt
    assert "SOURCE_ID:" in prompt
    result = json.loads((tmp_path / "Essay.citations.json").read_text())
    assert result["generation"] == {"provider": "codex", "model": "gpt-6-luna"}
    assert result["summary"]["mentions_linked"] == 2
    assert result["references"][0]["year"] == "2020"
    assert md.read_bytes() != original
    assert md.read_text().count("[Smith (2020)][cite-") == 2
    assert not (tmp_path / "Essay.cited.md").exists()
    assert not list(tmp_path.glob(".pdf-citations-*/"))


def test_codex_custom_model_override(tmp_path: Path, monkeypatch):
    md = source(tmp_path)
    set_codex_env(tmp_path, monkeypatch)
    assert service.find_citations(md, codex=True, model="gpt-6.1-sol") == 0
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[args.index("--model") + 1] == "gpt-6.1-sol"
    sidecar = json.loads((tmp_path / "Essay.citations.json").read_text())
    assert sidecar["generation"]["model"] == "gpt-6.1-sol"


def test_codex_not_installed_before_creating_workspace(tmp_path: Path, monkeypatch):
    md = source(tmp_path)
    monkeypatch.setattr(service.shutil, "which", lambda cmd: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Install it") as exc:
        service.find_citations(md, codex=True)
    assert "codex login" in str(exc.value)
    assert not list(tmp_path.glob(".pdf-citations-*"))


def test_codex_failure_keeps_logs_and_can_resume_without_cli(tmp_path: Path, monkeypatch):
    md = source(tmp_path)
    set_codex_env(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_EXIT", "1")
    with pytest.raises(RuntimeError, match="status 7"):
        service.find_citations(md, codex=True)
    work, = tmp_path.glob(".pdf-citations-*")
    assert "codex login" in (work / "codex-stderr.txt").read_text()
    assert json.loads((work / "source.json").read_text())["provider"] == "codex"
    assert not (work / "response.json").exists()
    monkeypatch.delenv("FAKE_CODEX_EXIT")
    monkeypatch.setattr(service.shutil, "which", lambda cmd: None)
    saved = {"schema_version":4,"source_id":citations.read_markdown(md)[2],
             "references":[{"l":6,"a":"Smith, A. (2020).","authors":["Smith, A."],"year":"2020"}],"mentions":["3|Smith (2020)"]}
    (work / "response.json").write_text(json.dumps(saved))
    assert service.find_citations(md, resume=work) == 0
    assert not work.exists()
    result = json.loads((tmp_path / "Essay.citations.json").read_text())
    assert result["generation"] == {"provider": "codex", "model": "gpt-6-luna"}


def test_resume_rejects_mismatched_model_and_provider(tmp_path: Path, monkeypatch):
    md = source(tmp_path)
    set_codex_env(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_EXIT", "1")
    with pytest.raises(RuntimeError):
        service.find_citations(md, codex=True)
    work, = tmp_path.glob(".pdf-citations-*")
    with pytest.raises(citations.CitationError, match="not 'gpt-6.1-sol'"):
        service.find_citations(md, resume=work, model="gpt-6.1-sol")
    with pytest.raises(ValueError, match="only to Gemini"):
        service.find_citations(md, codex=True, standard=True)
    assert work.exists()


def test_gemini_model_override_is_forwarded(tmp_path: Path, monkeypatch):
    md = source(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    seen = []
    def fake_gemini(request, workspace, *, standard, model):
        seen.append((standard,model))
        response = workspace / "response.json"
        response.write_text(json.dumps({
            "schema_version":4,"source_id":citations.read_markdown(md)[2],
            "references":[{"l":6,"a":"Smith, A. (2020).","authors":["Smith, A."],"year":"2020"}],"mentions":["3|Smith (2020)"]
        }))
        return response
    monkeypatch.setattr(service, "_gemini_citations", fake_gemini)
    assert service.find_citations(md, model="gemini-custom", standard=True) == 0
    assert seen == [(True,"gemini-custom")]
    result = json.loads((tmp_path / "Essay.citations.json").read_text())
    assert result["generation"] == {"provider": "gemini", "model": "gemini-custom"}
