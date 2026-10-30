from linguistic_alpha.config import REPO_ROOT, load_config
from linguistic_alpha.features import FEATURE_FAMILIES
from linguistic_alpha.models.specs import model_specs


def test_default_config_paths_resolve_inside_repo():
    cfg = load_config()
    assert cfg.paths.raw == REPO_ROOT / "data" / "raw"
    assert cfg["targets"]["car_windows"]["car_2_20"] == [2, 20]


def test_model_b_nests_model_a():
    specs = model_specs(load_config().raw)
    a, b = specs["A"], specs["B"]
    assert set(a.regressors) < set(b.regressors)
    assert a.target == b.target == "car_2_20"


def test_feature_families_registered_and_match_config():
    import linguistic_alpha.features.communication_state
    import linguistic_alpha.features.consistency
    import linguistic_alpha.features.interaction_quality  # noqa: F401

    cfg = load_config().raw["features"]
    assert set(FEATURE_FAMILIES) == {"communication_state", "interaction_quality", "consistency"}
    for name, family in FEATURE_FAMILIES.items():
        assert list(family.measures) == cfg[name]
