"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from ethirun.schemas import Recipe, Report
from tests.factories import make_recipe, make_report


@pytest.fixture
def sample_recipe() -> Recipe:
    return make_recipe()


@pytest.fixture
def sample_report() -> Report:
    return make_report()
