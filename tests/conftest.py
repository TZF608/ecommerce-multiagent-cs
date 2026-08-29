"""pytest 公共配置：把项目根目录加入 sys.path，并暴露假 LLM 夹具。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from tests.fake_llm import FakeChatModel, RaisingChatModel  # noqa: E402


@pytest.fixture
def fake_llm():
    return FakeChatModel()


@pytest.fixture
def raising_llm():
    return RaisingChatModel()
