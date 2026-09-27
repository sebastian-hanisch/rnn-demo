"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet, mit denselben Presets/Einstellungen wie im README zitiert."""
import rnn_constants as C
import rnn_evaluation as ev


def test_claim_t_sweep_success_rates():
    rows = ev.t_sweep()
    expected = {2: 1.0, 5: 1.0, 10: 1.0, 20: 13 / 15, 40: 0.6, 80: 1 / 3, 150: 0.4}
    for row in rows:
        assert round(row["rate"], 4) == round(expected[row["T"]], 4)


def test_claim_gradient_norm_first_step_decays_ten_orders_of_magnitude():
    rows = ev.gradient_norm_sweep()
    by_T = {r["T"]: r for r in rows}
    assert by_T[2]["norm_first"] > 1.0
    assert by_T[150]["norm_first"] < 1e-9
    # der letzte Zeitschritt bleibt in derselben Größenordnung über alle T
    for T in C.T_SWEEP_VALUES:
        assert 0.8 < by_T[T]["norm_last"] < 1.5


def test_claim_gradient_check_below_1e_minus_8():
    assert ev.gradient_check() < 1e-8


def test_claim_reduction_check_T1_is_exact():
    out = ev.reduction_check_T1()
    assert out["identical"]


def test_claim_preset_kurz_accuracy():
    p = C.PRESETS["kurz"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert round(out["test_accuracy"], 3) == 0.975


def test_claim_preset_lang_accuracy():
    p = C.PRESETS["lang"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert round(out["test_accuracy"], 2) == 0.30
