"""Shared test configuration.

The suite runs fully offline by *injecting* the test doubles directly: the fake
provider and the in-memory vector/conversation stores live under
``tests/doubles`` (WS10) and are wired into a pipeline via
``doubles.build_fake_pipeline`` (or constructed individually). Nothing is
selected through the production factories or via config strings, so there is no
environment mutation or settings cache to manage here. See ``test_retrieval.py``
/ ``test_pipeline_integration.py`` for the injection pattern.
"""

from __future__ import annotations
