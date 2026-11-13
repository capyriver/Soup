"""Tests for the optional Hugging Face Space hardware and ZeroGPU path (#2745)."""

from __future__ import annotations

import ast
import hashlib
from unittest.mock import MagicMock

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.deploy import render_space_template

runner = CliRunner()


def _deploy(monkeypatch, tmp_path, *arguments: str, api=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    if api is None:
        api = MagicMock()
    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: api)
    result = runner.invoke(
        app,
        [
            "deploy", "hf-space",
            "--model", "user/model",
            "--space", "user/space",
            "--yes",
            *arguments,
        ],
    )
    return result, api


def test_default_gradio_render_remains_byte_for_byte_unchanged() -> None:
    rendered = render_space_template("gradio-chat", model_repo="user/model")
    digests = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in rendered.items()
    }

    assert digests == {
        "app.py": "015681a39733aa13c89d721898a40971c7cae0564c726aede8be18a77daa9113",
        "requirements.txt": "1f17adad35e12f7c34f0ed2ce8b6bb5ef357901235051e11c29f1e13a984e18b",
        "README.md": "df71349c810f32183c228fcfef0535ddf85804dff6532c97a0a8d7cb0f670d54",
    }


def test_hardware_option_is_forwarded_to_hub(monkeypatch, tmp_path) -> None:
    result, api = _deploy(monkeypatch, tmp_path, "--hardware", "zero-a10g")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert api.create_repo.call_args.kwargs["space_hardware"] == "zero-a10g"


def test_default_hardware_option_is_omitted_from_hub_call(monkeypatch, tmp_path) -> None:
    result, api = _deploy(monkeypatch, tmp_path)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "space_hardware" not in api.create_repo.call_args.kwargs


def test_invalid_hardware_slug_fails_before_any_hub_call(monkeypatch, tmp_path) -> None:
    api = MagicMock()
    result, api = _deploy(monkeypatch, tmp_path, "--hardware", "Zero A10G", api=api)

    assert result.exit_code == 1
    assert "invalid --hardware" in result.output.lower()
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()


def test_zerogpu_refuses_builtin_streamlit_before_any_hub_call(monkeypatch, tmp_path) -> None:
    api = MagicMock()
    result, api = _deploy(
        monkeypatch,
        tmp_path,
        "--hardware", "zero-a10g",
        "--template", "streamlit-chat",
        api=api,
    )

    assert result.exit_code == 1
    assert "streamlit-chat" in result.output
    assert "ZeroGPU" in result.output
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()


def test_zerogpu_gradio_render_is_decorated_and_compiles() -> None:
    rendered = render_space_template("gradio-chat", model_repo="user/model", zero_gpu=True)
    app_source = rendered["app.py"]

    assert app_source.index("import spaces") < app_source.index("import gradio")
    assert "@spaces.GPU" in app_source
    assert "spaces" in rendered["requirements.txt"].splitlines()
    compile(ast.parse(app_source), "app.py", "exec")


def test_custom_template_is_uploaded_unchanged_with_hardware(monkeypatch, tmp_path) -> None:
    template = tmp_path / "custom"
    template.mkdir()
    source_files = {
        "app.py": "import gradio as gr\n",
        "README.md": "---\nsdk: gradio\n---\nCustom app\n",
        "requirements.txt": "gradio>=4\n",
    }
    for name, content in source_files.items():
        (template / name).write_text(content, encoding="utf-8")

    result, api = _deploy(
        monkeypatch,
        tmp_path,
        "--template-dir", "custom",
        "--hardware", "zero-a10g",
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert api.create_repo.call_args.kwargs["space_hardware"] == "zero-a10g"
    operations = api.create_commit.call_args.kwargs["operations"]
    assert {op.path_in_repo: op.path_or_fileobj for op in operations} == {
        name: content.encode("utf-8") for name, content in source_files.items()
    }
