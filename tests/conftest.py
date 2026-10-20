"""Shared test setup: the CardioLens /api routes need a key (D98)."""

import pytest

TEST_KEY = "test-key"
AUTH = {"X-API-Key": TEST_KEY}


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", f"tester:{TEST_KEY}")
