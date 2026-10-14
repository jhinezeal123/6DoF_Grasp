"""Cổng vào menu; chạy với stdlib, không import driver phần cứng."""

import sys
import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(source))

# Menu bootstrap độc lập với package robot (package robot cần Python 3.10).
# Vẫn dùng cùng source menu; không sao chép một menu Bash/Python thứ hai.
package_spec = importlib.util.spec_from_file_location(
    "_m750_menu",
    source / "m750" / "operator" / "__init__.py",
    submodule_search_locations=[str(source / "m750" / "operator")],
)
package = importlib.util.module_from_spec(package_spec)
sys.modules[package_spec.name] = package
package_spec.loader.exec_module(package)

from _m750_menu.console import main

if __name__ == "__main__":
    raise SystemExit(main())
