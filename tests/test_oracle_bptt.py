"""Unabhängige Orakel für BPTT und Training:

* Parametergradienten UND die Zeitschritt-Deltas dL/dz_t gegen den Complex-Step-Gradienten
  einer eigenen Vorwärtsrechnung (exakt bis auf Maschinengenauigkeit, anderer Rechenweg),
* `analyse` gegen einen eigenen Trainer mit matrixförmig neu geschriebenem BPTT
  (gleiche Fehlerzahlen je Epoche, gleiche Testgenauigkeit),
* die gemessenen Gradientennormen (README-Tabelle) gegen Complex-Step-Deltas.
"""
import numpy as np

import rnn_evaluation as ev
import rnn_model as M
import rnn_scenario as sc


def _score_c(p, seq, offsets=None):
    h = np.zeros(p["bh"].shape[0], dtype=complex)
    for t, x in enumerate(seq):
        z = p["Wxh"][:, 0] * x + p["Whh"] @ h + p["bh"]
        if offsets is not None:
            z = z + offsets[t]
        h = np.tanh(z)
    return p["Why"] @ h + p["by"]


def _cs_param_grads(p, seq, y):
    out = {}
    for key, v in p.items():
        v = np.asarray(v, dtype=float)
        g = np.zeros(v.shape)
        for idx in np.ndindex(*v.shape):
            q = {k: np.asarray(vv, dtype=complex).copy() for k, vv in p.items()}
            q[key][idx] += 1e-30j
            g[idx] = (-y * _score_c(q, seq)).imag / 1e-30
        out[key] = g
    return out


def _cs_deltas(p, seq, y):
    T, H = len(seq), p["bh"].shape[0]
    d = np.zeros((T, H))
    pc = {k: np.asarray(v, dtype=complex) for k, v in p.items()}
    for t in range(T):
        for j in range(H):
            off = np.zeros((T, H), dtype=complex)
            off[t, j] = 1e-30j
            d[t, j] = (-y * _score_c(pc, seq, off)).imag / 1e-30
    return d


def test_bptt_gradients_and_deltas_match_complex_step():
    rng = np.random.default_rng(11)
    for trial in range(40):
        H, T = int(rng.integers(1, 7)), int(rng.integers(1, 12))
        net = M.RNN(H, seed=trial)
        for k in net.params:
            net.params[k] = rng.normal(0, 0.8, size=net.params[k].shape)
        seq = rng.normal(0, 1, size=T)
        s, hs = net.forward(seq)
        assert abs(s - _score_c(net.params, seq).real) < 1e-12
        y = -1.0 if s >= 0 else 1.0
        grads, deltas = net.backward(y, s, hs, seq)
        ref = _cs_param_grads(net.params, seq, y)
        for k in grads:
            np.testing.assert_allclose(grads[k], ref[k], atol=1e-9)
        dref = _cs_deltas(net.params, seq, y)
        for t in range(T):
            np.testing.assert_allclose(deltas[t], dref[t], atol=1e-9)
        g2, d2 = net.backward(-y, s, hs, seq)  # korrekt klassifiziert: kein Gradient
        assert d2 is None and all(np.all(v == 0) for v in g2.values())


def test_gradient_norms_match_complex_step_deltas_including_vanishing_values():
    for T in (2, 10, 40, 150):
        d = ev.gradient_norms(T)
        net = M.RNN(8, seed=0)
        seq, _ = sc.make_sequence(np.random.default_rng(5), T)
        s = float(_score_c(net.params, seq).real)
        dr = _cs_deltas(net.params, seq, -np.sign(s) if s != 0 else 1.0)
        np.testing.assert_allclose(d["norm_first"], np.linalg.norm(dr[0]), rtol=1e-8)
        np.testing.assert_allclose(d["norm_last"], np.linalg.norm(dr[-1]), rtol=1e-8)


def _init(H, seed):
    r = np.random.default_rng(seed)
    lxh, lhh, lhy = np.sqrt(6.0 / (1 + H)), np.sqrt(6.0 / (2 * H)), np.sqrt(6.0 / (H + 1))
    return {"Wxh": r.uniform(-lxh, lxh, size=(H, 1)), "Whh": r.uniform(-lhh, lhh, size=(H, H)),
            "bh": np.zeros(H), "Why": r.uniform(-lhy, lhy, size=H), "by": np.zeros(())}


def _own_epoch(p, X, Y, eta):
    errs = 0
    for seq, y in zip(X, Y):
        T = len(seq)
        Hs = np.zeros((T + 1, p["bh"].shape[0]))
        for t in range(T):
            Hs[t + 1] = np.tanh(p["Wxh"][:, 0] * seq[t] + p["Whh"] @ Hs[t] + p["bh"])
        if y * (p["Why"] @ Hs[T] + p["by"]) > 0:
            continue
        errs += 1
        gx, gh, gb = np.zeros_like(p["Wxh"]), np.zeros_like(p["Whh"]), np.zeros_like(p["bh"])
        dh = -y * p["Why"]
        for t in range(T, 0, -1):
            dz = dh * (1 - Hs[t] ** 2)
            gx[:, 0] += dz * seq[t - 1]
            gh += np.outer(dz, Hs[t - 1])
            gb += dz
            dh = p["Whh"].T @ dz
        p["Why"] = p["Why"] - eta * (-y * Hs[T])
        p["by"] = p["by"] - eta * (-y)
        p["Wxh"], p["Whh"], p["bh"] = p["Wxh"] - eta * gx, p["Whh"] - eta * gh, p["bh"] - eta * gb
    return errs


def _own_acc(p, X, Y):
    ok = 0
    for seq, y in zip(X, Y):
        h = np.zeros(p["bh"].shape[0])
        for x in seq:
            h = np.tanh(p["Wxh"][:, 0] * x + p["Whh"] @ h + p["bh"])
        ok += (1.0 if p["Why"] @ h + p["by"] >= 0 else -1.0) == y
    return ok / len(Y)


def test_analyse_matches_independent_trainer():
    rng = np.random.default_rng(3)
    for _ in range(25):
        st = ev.Settings(T=int(rng.choice([2, 3, 5, 8, 10])), hidden=int(rng.integers(2, 9)),
                         n_train=int(rng.integers(20, 40)), n_test=20,
                         noise=float(rng.choice([0.0, 0.3, 0.6])),
                         eta=float(rng.choice([0.02, 0.1, 0.3])), epochs=int(rng.integers(3, 10)),
                         seed=int(rng.integers(0, 1000)))
        out = ev.analyse(st)
        tr = sc.make_dataset(st.n_train, st.T, st.seed, noise_std=st.noise)
        te = sc.make_dataset(st.n_test, st.T, st.seed + 1000, noise_std=st.noise)
        p = _init(st.hidden, st.seed)
        errs = [_own_epoch(p, tr.X, tr.y, st.eta) for _ in range(st.epochs)]
        assert errs == out["result"].errors_per_epoch
        assert abs(_own_acc(p, te.X, te.y) - out["test_accuracy"]) < 1e-12
