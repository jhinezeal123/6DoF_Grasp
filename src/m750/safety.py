"""Đường import cũ; implementation duy nhất nằm trong m750.robot.

Alias cùng module để import cũ, globals và monkeypatch vẫn dùng chung object.
"""
import sys
from .robot import safety as _implementation

sys.modules[__name__] = _implementation
