"""Khóa byte HTML trước refactor; AST tránh phải cài ROS để kiểm tra trang tĩnh."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    "limits, expected",
    [
        (
            (-3.141592653589793, 3.141592653589793),
            "dafd846115a032e49d4133950be7f36fb600a5c493e7c12e76efea522acdd1b1",
        ),
        ((-1.2, 0.8), "31e87603fa4f1dc1ccd786fb8beb55dfa72cdc717d64665ab324015481d9a994"),
    ],
)
def test_control_page_preserves_response_bytes(limits, expected):
    root = Path(__file__).resolve().parents[1]
    source = root / "src/m750/webui/web_control.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WebControlApp"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_html"
    )
    namespace = {"np": np, "json": json, "USB_CAMERAS": (("one", "/dev/fake-camera"),)}
    # Khi trang được tách ra, dùng chính renderer thật trong method facade.
    if (root / "src/m750/webui/pages/control.py").exists():
        namespace["render_control_page"] = importlib.import_module(
            "m750.webui.pages.control"
        ).render_html
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    robot = SimpleNamespace(
        JOINT_NAMES=tuple("q" + str(i) for i in range(1, 7)),
        rad_min=np.full(6, limits[0]),
        rad_max=np.full(6, limits[1]),
    )
    html = namespace["render_html"](SimpleNamespace(real_robot=robot))
    assert hashlib.sha256(html).hexdigest() == expected
