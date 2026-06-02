from __future__ import annotations


def test_package_imports() -> None:
    import dsl_elto

    assert hasattr(dsl_elto, "load_config")
    assert hasattr(dsl_elto, "validate_config")

