"""Registry and base-model integrity tests.

The highest-risk requirement is that the original weights (exp5_base.keras) are
never overwritten. These tests assert that registry operations leave the base
file byte-for-byte identical.
"""
import hashlib

from app.config import BASE_MODEL_PATH, CURRENT_MODEL_PATH
from app.services import registry


def _sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_base_and_current_are_separate_files():
    """Current must be a distinct file from base, so promoting never hits base."""
    assert BASE_MODEL_PATH != CURRENT_MODEL_PATH


def test_registry_update_does_not_touch_base(client):
    """A registry update + active-version change must not modify the base file."""
    import copy

    before = _sha256(BASE_MODEL_PATH)
    snapshot = copy.deepcopy(registry.get_registry())

    try:
        # Simulate registering a (fake) new version and promoting it.
        registry.update_registry(
            new_entry={"version": "v_test_integrity", "type": "retrained",
                       "samples_trained_on": 1,
                       "created_at": "2026-01-01T00:00:00+00:00",
                       "final_accuracy": 0.5, "final_loss": 1.0},
            active_version="v_test_integrity",
        )
        assert registry.get_active_version() == "v_test_integrity"

        after = _sha256(BASE_MODEL_PATH)
        assert before == after, "base model file was modified by a registry update"
    finally:
        # Fully restore the registry so no phantom version leaks into the app.
        registry._write_registry(snapshot)


def test_base_version_is_in_history(client):
    versions = [h["version"] for h in registry.get_history()]
    assert "v1_base" in versions


def test_find_unknown_version_returns_none(client):
    assert registry.find_version("does_not_exist") is None
