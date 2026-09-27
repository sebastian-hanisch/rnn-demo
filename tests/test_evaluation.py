import rnn_evaluation as ev


def test_analyse_short_T_reaches_high_accuracy():
    settings = ev.Settings(T=10, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.1,
                           epochs=30, seed=0)
    out = ev.analyse(settings)
    assert out["test_accuracy"] > 0.9


def test_t_sweep_short_sequences_always_succeed():
    rows = ev.t_sweep(values=(2, 5, 10), n_inits=8)
    for row in rows:
        assert row["rate"] == 1.0


def test_t_sweep_long_sequences_are_less_reliable_than_short():
    rows = ev.t_sweep(values=(5, 150), n_inits=15)
    rate_by_T = {r["T"]: r["rate"] for r in rows}
    assert rate_by_T[5] > rate_by_T[150]


def test_gradient_norm_sweep_first_step_norm_shrinks_monotonically_enough():
    rows = ev.gradient_norm_sweep(values=(2, 20, 80))
    norms_first = [r["norm_first"] for r in rows]
    assert norms_first[0] > norms_first[-1]
    assert norms_first[-1] < 1e-3
