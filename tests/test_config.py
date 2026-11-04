import pytest

from spectrafuse.config import DEFAULTS, make_settings


def test_defaults_match_paper():
    settings = make_settings()
    assert settings["chunk_tokens"] == 1200
    assert settings["overlap_tokens"] == 100
    assert settings["spectral_dims"] == 32
    assert settings["seed_count"] == 20
    assert settings["token_budget"] == 12000


def test_edge_weights_sum_to_one():
    assert sum(make_settings()["edge_weights"].values()) == pytest.approx(1.0)


def test_override_does_not_change_defaults():
    settings = make_settings({"chunk_tokens": 256})
    settings["edge_weights"]["semantic"] = 0.9
    assert settings["chunk_tokens"] == 256
    assert DEFAULTS["chunk_tokens"] == 1200
    assert DEFAULTS["edge_weights"]["semantic"] == 0.300


def test_unknown_setting_raises():
    with pytest.raises(KeyError):
        make_settings({"chunk_size": 100})
