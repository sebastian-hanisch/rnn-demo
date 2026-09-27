"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0
SIGNAL_POS = 0  # feste, frühe Position (erster Zeitschritt)

T_MIN, T_MAX, T_DEFAULT = 2, 200, 20
HIDDEN_MIN, HIDDEN_MAX, HIDDEN_DEFAULT = 2, 16, 8
N_TRAIN_MIN, N_TRAIN_MAX, N_TRAIN_DEFAULT = 20, 120, 60
N_TEST_MIN, N_TEST_MAX, N_TEST_DEFAULT = 20, 100, 40
NOISE_MIN, NOISE_MAX, NOISE_DEFAULT = 0.0, 0.6, 0.3
ETA_MIN, ETA_MAX, ETA_DEFAULT = 0.02, 0.3, 0.1
EPOCHS_MIN, EPOCHS_MAX, EPOCHS_DEFAULT = 10, 100, 30

# T-Sweep fuer den 📐-Abschnitt (Erfolgsquote ueber mehrere Zufalls-Inits)
T_SWEEP_VALUES = (2, 5, 10, 20, 40, 80, 150)
T_SWEEP_INITS = 15

PRESETS = {
    "kurz": dict(
        label="Kurze Sequenz — gelingt",
        T=10, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.1, epochs=30, seed=0,
        help="Bei kleinem T bleibt der Gradient bis zum ersten Zeitschritt stark genug - "
             "das Signal wird zuverlässig gelernt.",
    ),
    "lang": dict(
        label="Lange Sequenz — Erfolg wird unzuverlässig",
        T=150, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.1, epochs=30, seed=14,
        help="Bei großem T ist der Gradient am ersten Zeitschritt fast verschwunden - "
             "ob das Training gelingt, hängt stark von der zufälligen Initialisierung ab.",
    ),
}
