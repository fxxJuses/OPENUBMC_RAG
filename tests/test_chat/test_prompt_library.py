"""Tests for the PromptLibrary module."""

from pathlib import Path

import pytest

from ubmc_rag.prompts import PromptLibrary


@pytest.fixture
def lib():
    return PromptLibrary()


@pytest.fixture
def prompts_dir():
    return Path(__file__).parent.parent.parent / "ubmc_rag" / "prompts"


class TestLoad:
    def test_load_identity(self, lib):
        content = lib.load("chat/identity.md")
        assert "openUBMC" in content
        assert len(content) > 0

    def test_load_strategy(self, lib):
        content = lib.load("chat/strategy.md")
        assert "工作策略" in content

    def test_load_tools_guide(self, lib):
        content = lib.load("chat/tools_guide.md")
        assert "search_code" in content
        assert "search_multi" in content

    def test_load_evidence_rules(self, lib):
        content = lib.load("chat/evidence_rules.md")
        assert "证据约束" in content
        assert "Source" in content

    def test_load_domain_context(self, lib):
        content = lib.load("chat/domain_context.md")
        assert "openUBMC" in content
        assert "sensor" in content

    def test_load_output_format(self, lib):
        content = lib.load("chat/output_format.md")
        assert "回答格式" in content

    def test_load_caches_result(self, lib):
        a = lib.load("chat/identity.md")
        b = lib.load("chat/identity.md")
        assert a == b
        assert lib._cache.get("chat/identity.md") is not None

    def test_load_nonexistent_raises(self, lib):
        with pytest.raises(FileNotFoundError):
            lib.load("chat/nonexistent.md")


class TestCompose:
    def test_compose_two_files(self, lib):
        result = lib.compose("chat/identity.md", "chat/output_format.md")
        assert "openUBMC" in result
        assert "回答格式" in result
        # separator should be \n\n by default
        assert "\n\n" in result

    def test_compose_custom_separator(self, lib):
        result = lib.compose("chat/identity.md", "chat/output_format.md", separator="\n---\n")
        assert "\n---\n" in result


class TestGetSystemPrompt:
    def test_returns_nonempty(self, lib):
        prompt = lib.get_system_prompt()
        assert len(prompt) > 100

    def test_contains_all_sections(self, lib):
        prompt = lib.get_system_prompt()
        # identity
        assert "openUBMC 代码助手" in prompt
        # strategy
        assert "工作策略" in prompt
        # tools_guide
        assert "search_code" in prompt
        assert "search_multi" in prompt
        # evidence_rules
        assert "证据约束" in prompt
        # domain_context
        assert "微组件架构" in prompt
        # output_format
        assert "回答格式" in prompt

    def test_sections_in_order(self, lib):
        prompt = lib.get_system_prompt()
        # identity should come before strategy
        assert prompt.index("openUBMC 代码助手") < prompt.index("工作策略")
        # strategy before tools_guide
        assert prompt.index("工作策略") < prompt.index("工具使用指南")
        # evidence rules before domain context (use unique anchor from domain_context.md)
        assert prompt.index("证据约束") < prompt.index("## openUBMC 架构背景")
        # domain context before output format
        assert prompt.index("## openUBMC 架构背景") < prompt.index("回答格式")


class TestCustomDir:
    def test_custom_dir(self, tmp_path):
        # Create a fake prompt file
        (tmp_path / "test.md").write_text("hello world")
        lib = PromptLibrary(prompts_dir=tmp_path)
        assert lib.load("test.md") == "hello world"
