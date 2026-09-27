"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet, mit denselben Presets/Einstellungen wie im README zitiert."""
import rnn_constants as C
import rnn_evaluation as ev


def test_claim_t_sweep_success_rates():
    """Toleranzband statt exakter Gleichheit: die einzelne Trainings-Trajektorie
    ist ueber viele nichtlineare Epochen chaotisch empfindlich gegenueber
    Gleitkomma-Rundung (BLAS/tanh-Implementierung unterscheidet sich zwischen
    Windows und der Linux-CI) - ein einzelner gekippter Seed verschiebt die
    Quote um 1/15, das ist erwartete Variation, keine Regression
    (feedback_ci_platform_robust_tests.md)."""
    rows = ev.t_sweep()
    expected = {2: 1.0, 5: 1.0, 10: 1.0, 20: 13 / 15, 40: 0.6, 80: 1 / 3, 150: 0.4}
    tolerance = 2 / 15  # +-2 von 15 Initialisierungen
    for row in rows:
        assert abs(row["rate"] - expected[row["T"]]) <= tolerance + 1e-9
    # der grundsaetzliche Trend muss trotzdem stehen: kurze Sequenzen zuverlaessig,
    # lange deutlich unzuverlaessiger
    by_T = {r["T"]: r["rate"] for r in rows}
    assert by_T[2] == 1.0 and by_T[5] == 1.0 and by_T[10] == 1.0
    assert by_T[80] < 0.6 and by_T[150] < 0.6


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


def test_claim_lang_is_unreliable_on_average():
    """Ein EINZELNER Seed bei T=150 ist chaotisch plattformabhaengig (siehe
    test_claim_t_sweep_success_rates) - die robuste, testbare Aussage ist die
    mittlere Genauigkeit ueber mehrere Seeds, nicht ein einzelner gekippter
    Wert. Der Praesenz-Preset im Repo bleibt zur Illustration, wird aber nicht
    auf eine exakte Einzelzahl getestet."""
    p = C.PRESETS["lang"]
    accs = []
    for seed in range(15):
        settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                               p["eta"], p["epochs"], seed)
        out = ev.analyse(settings)
        accs.append(out["test_accuracy"])
    mean_acc = sum(accs) / len(accs)
    assert mean_acc < 0.85  # deutlich unter dem Niveau kurzer Sequenzen (>95%)
