"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, T-Sweep
(Erfolgsquote ueber Zufalls-Initialisierungen), Gradientennorm am ersten vs.
letzten Zeitschritt, Gradienten-Check, Korrektheits-Kette (T=1)."""
from dataclasses import dataclass

import numpy as np

import rnn_constants as C
import rnn_model as m
import rnn_scenario as sc


@dataclass(frozen=True)
class Settings:
    T: int
    hidden: int
    n_train: int
    n_test: int
    noise: float
    eta: float
    epochs: int
    seed: int


def analyse(settings: Settings) -> dict:
    train_ds = sc.make_dataset(settings.n_train, settings.T, settings.seed, noise_std=settings.noise)
    test_ds = sc.make_dataset(settings.n_test, settings.T, settings.seed + 1000,
                              noise_std=settings.noise)
    rnn = m.RNN(n_hidden=settings.hidden, seed=settings.seed)
    optimizer = m.SGD(settings.eta)
    result = m.train(rnn, optimizer, train_ds.X, train_ds.y, settings.epochs)
    return {
        "train_ds": train_ds, "test_ds": test_ds, "rnn": rnn, "result": result,
        "test_accuracy": m.accuracy(rnn, test_ds.X, test_ds.y),
    }


def t_sweep(
    values=C.T_SWEEP_VALUES, n_inits=C.T_SWEEP_INITS, n_train=C.N_TRAIN_DEFAULT,
    n_test=C.N_TEST_DEFAULT, noise=C.NOISE_DEFAULT, hidden=C.HIDDEN_DEFAULT,
    eta=C.ETA_DEFAULT, epochs=C.EPOCHS_DEFAULT,
) -> list:
    """Fuer jedes T: Anteil zufaelliger Initialisierungen, die auf >=90 %
    Testgenauigkeit kommen (fester Trainingsumfang, wie beim Hidden-Unit-Sweep
    in Stueck 2 - Erfolg ist bei grossem T nicht garantiert, sondern eine
    Frage der Initialisierung)."""
    rows = []
    for T in values:
        successes = 0
        for init_seed in range(n_inits):
            settings = Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                                eta=eta, epochs=epochs, seed=init_seed)
            out = analyse(settings)
            if out["test_accuracy"] >= 0.9:
                successes += 1
        rows.append({"T": T, "successes": successes, "n_inits": n_inits,
                      "rate": successes / n_inits})
    return rows


def gradient_norms(T: int, hidden: int = C.HIDDEN_DEFAULT, seed: int = 5) -> dict:
    """Norm des Gradienten am ersten (delta_1) vs. letzten (delta_T) Zeitschritt,
    an einem Punkt mit erzwungenem Fehler (sonst existiert kein Gradient)."""
    rnn = m.RNN(n_hidden=hidden, seed=0)
    rng = np.random.default_rng(seed)
    seq, _ = sc.make_sequence(rng, T)
    s, hs = rnn.forward(seq)
    y_forced = -np.sign(s) if s != 0 else 1.0
    grads, deltas = rnn.backward(y_forced, s, hs, seq)
    norm_first = float(np.linalg.norm(deltas[0]))
    norm_last = float(np.linalg.norm(deltas[-1]))
    return {"T": T, "norm_first": norm_first, "norm_last": norm_last,
            "ratio": norm_first / norm_last if norm_last else float("nan")}


def gradient_norm_sweep(values=C.T_SWEEP_VALUES, hidden: int = C.HIDDEN_DEFAULT) -> list:
    return [gradient_norms(T, hidden) for T in values]


def gradient_check(T: int = 6, hidden: int = 4, seed: int = 2, eps: float = 1e-5) -> float:
    """BPTT-Gradient gegen finite Differenzen."""
    rnn = m.RNN(n_hidden=hidden, seed=seed)
    rng = np.random.default_rng(1)
    seq, _ = sc.make_sequence(rng, T)
    s0, hs0 = rnn.forward(seq)
    y = -np.sign(s0) if s0 != 0 else 1.0
    grads, _ = rnn.backward(y, s0, hs0, seq)

    def loss() -> float:
        s, _ = rnn.forward(seq)
        return max(0.0, -y * s)

    max_rel_err = 0.0
    for name in ("Wxh", "Whh", "bh", "Why"):
        param = rnn.params[name]
        grad = grads[name]
        flat_param = param.reshape(-1)
        flat_grad = grad.reshape(-1)
        for idx in range(flat_grad.size):
            orig = flat_param[idx]
            flat_param[idx] = orig + eps
            l_plus = loss()
            flat_param[idx] = orig - eps
            l_minus = loss()
            flat_param[idx] = orig
            numeric = (l_plus - l_minus) / (2 * eps)
            analytic = float(flat_grad[idx])
            denom = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_err = max(max_rel_err, abs(numeric - analytic) / denom)
    return max_rel_err


def reduction_check_T1(hidden: int = 4, seed: int = 0, x_value: float = 0.7) -> dict:
    """T=1 hat keinen Rekursionsschritt (h_0=0) - reduziert sich strukturell
    auf eine einzelne dichte Schicht mit tanh-Aktivierung."""
    rnn = m.RNN(n_hidden=hidden, seed=seed)
    seq = np.array([x_value])
    s, _ = rnn.forward(seq)
    h1_manual = np.tanh(rnn.params["Wxh"][:, 0] * seq[0] + rnn.params["bh"])
    s_manual = float(rnn.params["Why"] @ h1_manual + rnn.params["by"])
    return {"s_rnn": s, "s_dense": s_manual, "identical": bool(np.isclose(s, s_manual))}
