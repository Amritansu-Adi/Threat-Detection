import os
import sys

# Ensure repo root is on sys.path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest


@pytest.fixture(scope='session')
def fixtures_dir():
    return os.path.join(REPO_ROOT, 'tests', 'fixtures')
