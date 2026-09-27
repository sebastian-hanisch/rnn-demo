from streamlit.testing.v1 import AppTest


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=60)
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
    assert metrics["Testgenauigkeit"] in ("98%", "97%", "100%")


def test_preset_lang_shows_low_accuracy():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Lange Sequenz — Erfolg wird unzuverlässig"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Testgenauigkeit"] == "30%"


def test_T_slider_extreme_values_do_not_crash():
    at = _fresh()
    t_slider = [s for s in at.slider if s.label.startswith("Sequenzlänge")][0]
    t_slider.set_value(t_slider.min).run()
    assert not at.exception
    t_slider = [s for s in at.slider if s.label.startswith("Sequenzlänge")][0]
    t_slider.set_value(t_slider.max).run()
    assert not at.exception
