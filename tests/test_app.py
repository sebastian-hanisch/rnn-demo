from streamlit.testing.v1 import AppTest

# Hoeherer Timeout als in den Geschwister-Repos: der T-Sweep (15 Inits x 7
# T-Werte bis T=150) braucht auf einem langsamen CI-Runner deutlich laenger
# als lokal (siehe rnn_evaluation.t_sweep-Performance-Hinweis in der Memory-
# Notiz); 60s reichte auf der Linux-CI nicht.
_APP_TIMEOUT = 180


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Kontakt aufnehmen" in c for c in captions)


def test_preset_kurz_shows_high_accuracy():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Kurze Sequenz — gelingt"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert float(metrics["Testgenauigkeit"].rstrip("%")) >= 90.0


def test_preset_lang_runs_and_reports_a_percentage():
    """Keine exakte Prozentzahl hier: ein einzelner Seed bei T=150 ist ueber
    Plattformen hinweg chaotisch (siehe test_claims.py). Nur pruefen, dass die
    App laeuft und eine gueltige Prozentzahl anzeigt."""
    at = _fresh()
    btn = [b for b in at.button if b.label == "Lange Sequenz — Erfolg wird unzuverlässig"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert 0.0 <= float(metrics["Testgenauigkeit"].rstrip("%")) <= 100.0


def test_T_slider_extreme_values_do_not_crash():
    at = _fresh()
    t_slider = [s for s in at.slider if s.label.startswith("Sequenzlänge")][0]
    t_slider.set_value(t_slider.min).run()
    assert not at.exception
    t_slider = [s for s in at.slider if s.label.startswith("Sequenzlänge")][0]
    t_slider.set_value(t_slider.max).run()
    assert not at.exception
