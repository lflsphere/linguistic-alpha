from linguistic_alpha.config import REPO_ROOT, load_config
from linguistic_alpha.features import FEATURE_FAMILIES
from linguistic_alpha.models.specs import model_specs


def test_default_config_paths_resolve_inside_repo():
    cfg = load_config()
    assert cfg.paths.raw == REPO_ROOT / "data" / "raw" / "sp500_pit_2008"
    assert cfg["targets"]["car_windows"]["car_2_20"] == [2, 20]


def test_model_b_nests_model_a():
    specs = model_specs(load_config().raw)
    a, b = specs["A"], specs["B"]
    assert set(a.regressors) < set(b.regressors)
    assert a.target == b.target == "car_2_20"


def test_model_c_nests_model_b_and_adds_only_finbert():
    cfg = load_config().raw
    specs = model_specs(cfg)
    b, c = specs["B"], specs["C"]
    assert set(b.regressors) < set(c.regressors)
    assert set(c.regressors) - set(b.regressors) == set(cfg["models"]["measures"]["finbert"])


def test_c_alone_is_a_plus_finbert_and_comparisons_flag_nesting():
    from linguistic_alpha.models.specs import COMPARISONS
    cfg = load_config().raw
    s = model_specs(cfg)
    m = cfg["models"]["measures"]
    assert set(s["C_alone"].regressors) == (set(s["A"].regressors) | set(m["finbert"])
                                            | set(m["finbert_interactions"]))
    for base, alt, nested in COMPARISONS:
        assert nested == (set(s[base].regressors) < set(s[alt].regressors))


def test_every_family_measure_is_assigned_to_one_method():
    cfg = load_config().raw
    by_method = cfg["models"]["measures"]
    family = {m for k, v in cfg["features"].items() if isinstance(v, list) and k != "future" for m in v}
    assigned = set(by_method["dictionary"]) | set(by_method["later"])
    assert family == assigned
    assert not set(by_method["dictionary"]) & set(by_method["later"])


def test_feature_families_registered_and_match_config():
    import linguistic_alpha.features.communication_state
    import linguistic_alpha.features.consistency
    import linguistic_alpha.features.interaction_quality  # noqa: F401

    cfg = load_config().raw["features"]
    assert set(FEATURE_FAMILIES) == {"communication_state", "interaction_quality", "consistency"}
    for name, family in FEATURE_FAMILIES.items():
        assert list(family.measures) == cfg[name]


def test_release_metrics_in_a_and_interactions_only_in_b_and_c():
    cfg = load_config().raw
    s = model_specs(cfg)
    rel = set(cfg["controls"]["release_metrics"])
    inter = set(cfg["models"]["measures"]["interactions"])
    assert rel <= set(s["A"].regressors)
    fb_inter = set(cfg["models"]["measures"]["finbert_interactions"])
    assert not inter & set(s["A"].regressors) and not inter & set(s["C_alone"].regressors)
    assert inter <= set(s["B"].regressors) and inter <= set(s["C"].regressors)
    assert fb_inter <= set(s["C_alone"].regressors) and not fb_inter & set(s["C"].regressors)


def test_every_model_has_both_reaction_legs():
    from linguistic_alpha.models.specs import model_specs_by_leg
    cfg = load_config().raw
    legs = model_specs_by_leg(cfg)
    assert set(legs) == {"main", "comparison"}
    for name in ("A", "B", "C", "C_alone"):
        main, comp = set(legs["main"][name].regressors), set(legs["comparison"][name].regressors)
        assert main - comp == {"car_m1_1"} and comp - main == {"car_0_1"}
