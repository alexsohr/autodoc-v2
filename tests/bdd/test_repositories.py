"""BDD tests for repository management API

This module links the repositories.feature file to its step definitions.
"""

import pytest
from pytest_bdd import scenarios

# Import all step definitions
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.repository_steps import *  # noqa: F401, F403

# Load scenarios from feature file
scenarios("features/repositories.feature")

