import numpy as np

import rnn_evaluation as ev
import rnn_model as m
import rnn_scenario as sc


def test_forward_returns_scalar_score():
    rnn = m.RNN(n_hidden=4, seed=0)
    seq = np.zeros(10)
    s, hs = rnn.forward(seq)
    assert isinstance(s, float)
    assert len(hs) == 11  # h_0 .. h_10


def test_backward_returns_none_deltas_when_correctly_classified():
    rnn = m.RNN(n_hidden=4, seed=0)
    rnn.params["Why"] = np.zeros(4)
    rnn.params["by"] = np.asarray(5.0)  # s wird konstant 5.0, unabhaengig vom Zustand
    seq = np.zeros(5)
    s, hs = rnn.forward(seq)
    grads, deltas = rnn.backward(1.0, s, hs, seq)  # y*s = 5 > 0
    assert deltas is None
    assert np.allclose(grads["Whh"], 0.0)


def test_backward_deltas_ordered_first_to_last():
    rnn = m.RNN(n_hidden=4, seed=1)
    rng = np.random.default_rng(2)
    seq, _ = sc.make_sequence(rng, T=8)
    s, hs = rnn.forward(seq)
    y = -np.sign(s) if s != 0 else 1.0
    grads, deltas = rnn.backward(y, s, hs, seq)
    assert deltas is not None
    assert len(deltas) == 8  # ein Delta je Zeitschritt


def test_gradient_check_below_1e_minus_8():
    assert ev.gradient_check() < 1e-8


def test_gradient_norm_decays_with_T():
    row_short = ev.gradient_norms(T=5)
    row_long = ev.gradient_norms(T=100)
    assert row_long["norm_first"] < row_short["norm_first"]
    # der letzte Zeitschritt bleibt deutlich stabiler als der erste
    assert row_long["norm_last"] > row_long["norm_first"] * 10


def test_reduction_check_T1_is_exact():
    out = ev.reduction_check_T1()
    assert out["identical"]


def test_training_reduces_errors():
    ds = sc.make_dataset(30, T=10, seed=1)
    rnn = m.RNN(n_hidden=8, seed=0)
    result = m.train(rnn, m.Adam(0.1), ds.X, ds.y, epochs=20)
    assert result.errors_per_epoch[-1] <= result.errors_per_epoch[0]


def test_short_sequences_are_almost_always_learnable():
    ds_train = sc.make_dataset(60, T=5, seed=1)
    ds_test = sc.make_dataset(40, T=5, seed=2)
    rnn = m.RNN(n_hidden=8, seed=0)
    m.train(rnn, m.Adam(0.1), ds_train.X, ds_train.y, epochs=30)
    assert m.accuracy(rnn, ds_test.X, ds_test.y) > 0.9
