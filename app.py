"""RNN — Zustand über die Zeit

Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der "Neuronale Netze"-Reihe der "Konzepte"-Reihe:
Perceptron -> MLP+Backpropagation -> {CNN, RNN -> LSTM -> Attention/Transformer}.
Ein MLP behandelt jedes Beispiel unabhängig - ein RNN trägt einen Zustand über
eine Sequenz. Hier zum ersten Mal: Vehikel C, eine Signal-in-Rauschen-Sequenz
der Länge T, an der auch LSTM und Attention/Transformer später gemessen werden.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import rnn_constants as C
import rnn_evaluation as ev
import rnn_model as m
import rnn_presets as pr
import rnn_scenario as sc
import rnn_visualization as viz

st.set_page_config(page_title="RNN", layout="wide")


@st.cache_data(show_spinner=False)
def _analyse(T, hidden, n_train, n_test, noise, eta, epochs, seed):
    settings = ev.Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                           eta=eta, epochs=epochs, seed=seed)
    out = ev.analyse(settings)
    return {"test_accuracy": out["test_accuracy"],
            "errors_per_epoch": out["result"].errors_per_epoch}


@st.cache_data(show_spinner=False)
def _t_sweep():
    return ev.t_sweep()


@st.cache_data(show_spinner=False)
def _gradient_norm_sweep():
    return ev.gradient_norm_sweep()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


@st.cache_data(show_spinner=False)
def _reduction_check():
    out = ev.reduction_check_T1()
    return out["s_rnn"], out["s_dense"], out["identical"]


st.title("🧠 RNN — Zustand über die Zeit")
st.markdown(
    "Ein MLP (Stück 2) behandelt jedes Beispiel unabhängig. Ein RNN trägt einen **Zustand** "
    "$h_t$ über eine Sequenz - jeder Zeitschritt aktualisiert denselben Zustand mit denselben "
    "(geteilten) Gewichten. Das erlaubt Aufgaben, bei denen die Antwort von etwas abhängt, das "
    "weit zurückliegt. Genau dort liegt auch das Problem: der Gradient, der dieses "
    "'weit zurückliegend' lernen soll, wird auf dem Weg dorthin immer schwächer."
)
st.caption(
    "Stück 4 (Geschwister von CNN) der 'Neuronale Netze'-Reihe. Geplante Folgestücke "
    "(noch nicht gebaut): LSTM, Attention/Transformer - dieselbe Aufgabe, größeres T."
)

with st.expander("So funktioniert das RNN", expanded=True):
    st.markdown(
        "1. Zustand: $h_t=\\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$ - dieselben Gewichte zu jedem "
        "Zeitschritt (Gewichtsteilung über die Zeit, wie die Faltung Gewichtsteilung über den "
        "Raum nutzte).\n"
        "2. Ausgabe nur am LETZTEN Zeitschritt: $s=W_{hy}h_T+b_y$, Perceptron-Criterion-Verlust.\n"
        "3. Rückwärtspass (Backpropagation Through Time): der Fehler wird rekursiv durch die "
        "Zeit zurückgereicht, ein $\\tanh'$-Faktor pro Schritt - genau das lässt den Gradienten "
        "am ersten Zeitschritt mit wachsendem T verschwinden."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    T = st.slider("Sequenzlänge T", C.T_MIN, C.T_MAX, ss["T"], key="widget_T",
                 on_change=pr.store_from_widget, args=("T",))
    ss["T"] = T
    hidden = st.slider("Verdeckte Einheiten", C.HIDDEN_MIN, C.HIDDEN_MAX, ss["hidden"],
                       key="widget_hidden", on_change=pr.store_from_widget, args=("hidden",))
    ss["hidden"] = hidden
    n_train = st.slider("Trainingsbeispiele", C.N_TRAIN_MIN, C.N_TRAIN_MAX, ss["n_train"],
                        step=10, key="widget_n_train", on_change=pr.store_from_widget,
                        args=("n_train",))
    ss["n_train"] = n_train
    noise = st.slider("Rauschanteil", C.NOISE_MIN, C.NOISE_MAX, ss["noise"], step=0.05,
                      key="widget_noise", on_change=pr.store_from_widget, args=("noise",))
    ss["noise"] = noise
    eta = st.slider("Lernrate η", C.ETA_MIN, C.ETA_MAX, ss["eta"], step=0.02,
                    key="widget_eta", on_change=pr.store_from_widget, args=("eta",))
    ss["eta"] = eta
    epochs = st.slider("Epochen", C.EPOCHS_MIN, C.EPOCHS_MAX, ss["epochs"], step=5,
                       key="widget_epochs", on_change=pr.store_from_widget, args=("epochs",))
    ss["epochs"] = epochs
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)
    n_test = C.N_TEST_DEFAULT

pr.sync_query_params(dict(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                         eta=eta, epochs=epochs, seed=seed))

st.markdown("---")
st.subheader("📈 Eine Beispielsequenz")
rng_preview = np.random.default_rng(int(seed) + 5000)
seq_preview, label_preview = sc.make_sequence(rng_preview, T, noise_std=noise)
st.plotly_chart(viz.build_sequence_figure(seq_preview, label_preview),
                key=f"seq_{T}_{noise}_{seed}", use_container_width=True)

with st.spinner("Trainiere RNN..."):
    out = _analyse(T, hidden, n_train, n_test, noise, eta, epochs, int(seed))

st.subheader("🎯 Was am Ende steht")
m1, m2 = st.columns(2)
m1.metric("Testgenauigkeit", f"{out['test_accuracy']*100:.0f}%")
m2.metric("Sequenzlänge T", f"{T}")
st.plotly_chart(viz.build_error_curve_figure(out["errors_per_epoch"]),
                key=f"errcurve_{T}_{hidden}_{n_train}_{noise}_{eta}_{epochs}_{seed}",
                use_container_width=True)

st.markdown("---")
st.subheader("🎯 Erfolgsquote vs. Sequenzlänge (fester Trainingsumfang)")
with st.spinner("Berechne Erfolgsquote über 15 Zufalls-Initialisierungen je Sequenzlänge "
               "(einmalig, kann bis zu ~40 Sekunden dauern - lange Sequenzen sind langsam "
               "zu trainieren)..."):
    sweep = _t_sweep()
st.plotly_chart(viz.build_t_sweep_figure(sweep), key="t_sweep_chart", use_container_width=True)
st.caption(
    "Bei T=2–10 gelingt fast jede Initialisierung (100 %). Ab T=20 wird der Erfolg "
    "zunehmend eine Frage des Zufalls (Initialisierung) statt der Garantie - bei T=150 nur "
    "noch rund die Hälfte."
)

st.subheader("🎯 Gradientennorm: erster vs. letzter Zeitschritt")
grad_rows = _gradient_norm_sweep()
st.plotly_chart(viz.build_gradient_norm_figure(grad_rows), key="grad_norm_chart",
                use_container_width=True)
st.caption(
    "Die Gradientennorm am LETZTEN Zeitschritt bleibt ungefähr konstant - am ERSTEN "
    "Zeitschritt fällt sie um mehrere Zehnerpotenzen, je größer T wird. Das ist das "
    "eigentliche 'Verschwinden', nicht nur ein Symptom in der Genauigkeit."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Ein Bit reicht als Erinnerung | Komplexere Muster (z. B. zwei Positionen in "
    "Beziehung setzen) sind noch schwerer über die Zeit zu tragen | — |\n"
    "| Fester Trainingsumfang (Epochen) | Mit unbegrenzt vielen Epochen konvergiert das RNN "
    "auch bei großem T oft noch - das Problem ist praktische Trainierbarkeit, nicht absolute "
    "Unmöglichkeit | — |\n"
    "| Gradient muss weit zurückreichen | Gatter, die den Gradienten gezielt durchlassen, "
    "beheben genau das | LSTM (nächstes Stück) |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Rückwärtspass (BPTT):** bei Fehlklassifikation ($ys\le0$): $\delta_T=(W_{hy}\cdot(-y))\odot
(1-h_T^2)$, dann rekursiv für $t=T-1,\dots,1$: $\delta_t=(W_{hh}^\top\delta_{t+1})\odot(1-h_t^2)$.

**Korrektheits-Kette (T=1):** ohne Rekursionsschritt ist $h_1=\tanh(W_{xh}x_1+b_h)$ (da
$h_0=0$) - das RNN reduziert sich auf eine einzelne dichte Schicht.
"""
    )
    s_rnn, s_dense, identical = _reduction_check()
    c1, c2 = st.columns(2)
    c1.metric("RNN-Ausgabe (T=1)", f"{s_rnn:.6f}")
    c2.metric("Dicht nachgebaut", f"{s_dense:.6f}")
    st.metric("Identisch?", "Ja" if identical else "Nein (Fehler!)")
    st.markdown("**Gradienten-Check** (gegen finite Differenzen):")
    st.metric("Maximaler relativer Fehler", f"{_gradient_check():.2e}")
    st.markdown(
        "**Literatur:** Elman, J. L. (1990). *Finding Structure in Time.* Cognitive Science, "
        "14(2), 179–211. — Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* "
        "Neural Computation, 9(8), 1735–1780 (Adding Problem/Temporal Order als Vehikel-Vorlage)."
    )
    st.caption(
        "Implementiert in `rnn_model.py` (RNN, Adam), `rnn_scenario.py` (Sequenzen), "
        "`rnn_evaluation.py` (T-Sweep, Gradientennorm, Gradienten-Check, Reduktions-Check), "
        "`rnn_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
