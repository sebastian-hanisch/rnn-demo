import rnn_constants as C
import rnn_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert C.T_MIN <= preset["T"] <= C.T_MAX
        assert C.HIDDEN_MIN <= preset["hidden"] <= C.HIDDEN_MAX


def test_preset_kurz_succeeds():
    p = C.PRESETS["kurz"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert out["test_accuracy"] > 0.9


def test_preset_lang_has_valid_settings_and_produces_a_probability():
    """Nur ein Rauchtest hier: der einzelne konfigurierte Seed ist ueber
    Plattformen hinweg chaotisch (siehe test_claim_lang_is_unreliable_on_average
    in test_claims.py fuer die robuste, gemittelte Aussage)."""
    p = C.PRESETS["lang"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert 0.0 <= out["test_accuracy"] <= 1.0
