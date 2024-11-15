import ast
import pathlib
import sys
from argparse import ArgumentParser


def is_standard_library(module_name: str) -> bool:
    """Check if a module is part of the Python standard library."""
    return module_name in sys.builtin_module_names


def extract_imports_from_file(file_path: pathlib.Path) -> set[str]:
    """Extract imported modules from a Python file."""
    with file_path.open("r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]  # Top-level module
                imports.add(module_name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module.split(".")[0]  # Top-level module
            imports.add(module_name)
    return imports


def list_external_packages(module_path: pathlib.Path) -> set[str]:
    """List all external packages used in a module."""
    all_imports = set()
    for py_file in module_path.rglob("*.py"):
        all_imports.update(extract_imports_from_file(py_file))

    # Get all internal modules based on the directory structure
    internal_modules = {
        module_path.relative_to(module_path.parent).as_posix().replace("/", ".")
    }
    for py_file in module_path.rglob("*.py"):
        relative_path = py_file.relative_to(module_path.parent).as_posix()
        module_name = relative_path.replace(".py", "").replace("/", ".")
        internal_modules.add(module_name.split(".")[1])  # Add top-level modules

    # Filter out standard library and internal modules
    external_packages = {
        pkg
        for pkg in all_imports
        if not is_standard_library(pkg) and pkg not in internal_modules
    }
    return external_packages


def main(module: str) -> None:
    """Main function to list external packages."""
    module_path = pathlib.Path(module).resolve()
    if not module_path.is_dir():
        print(f"Error: Module path {module_path} is not a directory.")
        sys.exit(1)

    external_packages = list_external_packages(module_path)
    print("External packages used in the module:")
    print("[")
    print(",\n".join(f'"{pkg}"' for pkg in sorted(external_packages)))
    print("]")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="List external packages used in a Python module."
    )
    parser.add_argument(
        "--module",
        type=str,
        default="src/coastpy",
        help="Path to the module directory (default: src/coastpy).",
    )
    args = parser.parse_args()
    main(args.module)
