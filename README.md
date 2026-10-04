# RNN – Zustand über die Zeit – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-rnn-demo.streamlit.app/)**

Stück 4 der **Neuronale-Netze-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Ein MLP (Stück 2) behandelt jedes Beispiel unabhängig. Ein RNN (Elman 1990) trägt einen
**Zustand** über eine Sequenz – jeder Zeitschritt aktualisiert denselben Zustand mit denselben
(geteilten) Gewichten. Hier zum ersten Mal: **Vehikel C**, eine Signal-in-Rauschen-Sequenz der
Länge $T$, die auch die nächsten beiden Stücke (LSTM, Attention/Transformer) tragen wird.

**Einordnung in die Reihe:**

```
Perceptron (WURZEL)                              [gebaut]
 └─ MLP + Backpropagation                        [gebaut]
      ├─ CNN                                     [gebaut]
      └─ RNN                                     [DIESES STÜCK]
           └─ LSTM                               [gebaut]
                └─ Attention/Transformer         [gebaut]
```

**Ergebnis in Kürze:** Ein Bit Information über eine Sequenz zu tragen gelingt bei kurzen
Sequenzen fast immer (100 % Erfolgsquote bei $T\le10$), wird aber mit wachsendem $T$ zunehmend
eine Frage des Zufalls (nur noch 33–40 % bei $T\ge80$) – nicht weil das RNN es *nicht könnte*,
sondern weil der Gradient, der das lernen soll, unterwegs praktisch verschwindet: die
Gradientennorm am ERSTEN Zeitschritt fällt von $\approx1{,}2$ bei $T=2$ auf $6{,}7\cdot10^{-11}$
bei $T=150$ (über 10 Zehnerpotenzen), während sie am LETZTEN Zeitschritt konstant bei $\approx1$
bleibt.

## Warum dieses Problem

Ein MLP (Stück 2) hat keinen Begriff von "vorher" – jedes Beispiel ist unabhängig. Viele echte
Aufgaben (Sprache, Zeitreihen, Sensordaten) brauchen aber genau das: eine Antwort, die von
etwas abhängt, das weiter zurückliegt. Ein RNN trägt einen Zustand über die Zeit – aber
Backpropagation Through Time zeigt sofort das Grundproblem, das LSTM (nächstes Stück) beheben
wird.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Genauigkeit bricht ab einer Sequenzlänge $T$ zusammen | ⚠️ Bei ausreichend Trainingsepochen (100) konvergiert das RNN auch bei $T=80$ noch auf 100 % – das Problem ist praktische Trainierbarkeit in einem realistischen Budget, nicht absolute Unmöglichkeit (siehe Plan-Korrektur unten) |
| Bei begrenztem, realistischem Trainingsbudget wird der Erfolg mit wachsendem $T$ unzuverlässiger | ✅ Erfolgsquote über 15 Zufalls-Initialisierungen: 100 % bei $T\le10$, fällt auf 33–40 % bei $T\ge80$ |
| Gradientennorm am ersten Zeitschritt fällt mit wachsendem $T$ gegen null, am letzten bleibt sie stabil | ✅ $\|\delta_1\|$ fällt von 1,16 ($T=2$) auf $6{,}7\cdot10^{-11}$ ($T=150$); $\|\delta_T\|$ bleibt zwischen 0,90 und 1,27 |
| Gradienten-Check gegen finite Differenzen unter $10^{-8}$ | ✅ 2,75·10⁻⁹ |
| RNN mit $T=1$ reduziert sich exakt auf eine dichte Schicht | ✅ identische Ausgabe (`test_claim_reduction_check_T1_is_exact`) |
| ⚠️ Plan-Korrektur: Adam als Standard-Optimierer verwässerte den Effekt | ⚠️ Mit Adam (adaptive Normalisierung pro Parameter) konvergierten viele lange Sequenzen trotzdem zuverlässig – Adams Division durch die eigene historische Gradientengröße kompensiert einen Teil des verschwindenden Gradienten. Umgestellt auf einfaches SGD ohne Normalisierung, damit die Demo den Effekt selbst zeigt statt ihn durch die Optimierer-Wahl zu verdecken. |

## Befunde (gemessen, keine Behauptungen)

**Erfolgsquote vs. Sequenzlänge** (15 Zufalls-Initialisierungen je $T$, festes Budget: 60
Trainingsbeispiele, 30 Epochen, SGD):

| T | Erfolgsquote |
|---|---|
| 2 | 100 % |
| 5 | 100 % |
| 10 | 100 % |
| 20 | 87 % |
| 40 | 60 % |
| 80 | 33 % |
| 150 | 40 % |

**Gradientennorm am ersten vs. letzten Zeitschritt:**

| T | $\|\delta_1\|$ | $\|\delta_T\|$ |
|---|---|---|
| 2 | 1,16 | 1,27 |
| 5 | 1,69 | 1,22 |
| 10 | 0,35 | 1,04 |
| 20 | 0,085 | 0,92 |
| 40 | 0,00029 | 1,01 |
| 80 | 0,0000108 | 0,90 |
| 150 | 0,000000000067 | 0,98 |

**Presets:** "Kurze Sequenz — gelingt" ($T=10$): 97,5 % Testgenauigkeit. "Lange Sequenz — Erfolg
wird unzuverlässig" ($T=150$): über 15 Seeds gemittelt nur **≈70 % Testgenauigkeit** (Streuung
30–100 % je nach Seed) – deutlich unter dem Niveau kurzer Sequenzen und mit erheblicher
Schwankung. Der einzelne im Preset gezeigte Seed liefert je nach Rechner/Plattform eine andere
konkrete Zahl (siehe Plan-Korrektur unten) – die robuste Aussage ist der Mittelwert über viele
Seeds, nicht ein einzelner Lauf.

## Modell und Verfahren

- `rnn_scenario.py` – Signal-in-Rauschen-Sequenzgenerator (Vehikel C).
- `rnn_model.py` – `RNN` (Elman, BPTT von Hand), `SGD`, `Adam`.
- `rnn_evaluation.py` – $T$-Sweep (Erfolgsquote), Gradientennorm-Messung, Gradienten-Check,
  Korrektheits-Kette ($T=1$).
- `rnn_visualization.py` – Plotly: Sequenz-Anzeige, Erfolgsquote-vs-$T$, Gradientennorm-vs-$T$.

## Was die App zeigt

Sequenzlänge, verdeckte Einheiten, Trainingsgröße, Rauschen, Lernrate, Epochen und Seed in der
Sidebar; eine Beispielsequenz mit markiertem Signal; Fehlerkurve und Testgenauigkeit für die
aktuelle Konfiguration; der Erfolgsquote-vs-$T$-Sweep (auf Klick, einmalig bis zu ~40 Sekunden)
und die Gradientennorm-vs-$T$-Kurve als zentrale Belege; ein "📐"-Abschnitt mit der
$T=1$-Korrektheits-Kette und dem Gradienten-Check.

## Was nicht funktioniert hat / Grenzen

**Echte Plan-Korrektur (siehe Vorab-Hypothesen):** Die erste Version nutzte Adam als
Standard-Optimierer (wie in den Geschwister-Repos). Das verdeckte den eigentlichen Effekt: mit
genug Epochen (100) erreichte das RNN mit Adam auch bei $T=80$ zuverlässig 100 % Testgenauigkeit
– Adams Division durch die eigene historische Gradientengröße normalisiert kleine Gradienten
relativ auf, was den verschwindenden Gradienten teilweise kompensiert. Umgestellt auf einfaches
SGD (ohne adaptive Normalisierung) bei einem realistischen, festen Epochenbudget – seither zeigt
sich der Effekt klar und robust über mehrere Seeds.

**Zweite echte Plan-Korrektur (CI-Lauf, Push von Stück 4):** Die erste Testversion pinnte die
Testgenauigkeit eines einzelnen, gezielt gewählten Seeds bei $T=150$ exakt auf 30 % – das
Training über 30 Epochen auf dieser nichtlinearen $\tanh$-Dynamik ist chaotisch genug, dass
winzige Gleitkomma-Unterschiede zwischen Windows und der Linux-CI (unterschiedliche
BLAS-Implementierung) nach vielen Schritten zu einem völlig anderen Ergebnis führen (auf der CI
85 % statt 30 % für denselben Seed). Behoben durch ein Toleranzband beim $T$-Sweep und eine über
15 Seeds gemittelte, robuste Aussage für den "Lange Sequenz"-Fall statt einer einzelnen exakten
Zahl (siehe `feedback_ci_platform_robust_tests.md` in der Projekt-Historie – dieselbe Lehre, die
bereits vor dieser Linie bekannt war, hier aber zunächst nicht angewendet wurde).

**Dritte Korrektur (Keep-alive-Check auf sebastianhanisch.net, 2026-09-27):** Der $T$-Sweep lief
zunächst automatisch bei jedem Laden der Seite. Nach dem Einschlafen der App auf Streamlit Cloud
addierte sich die Aufweckzeit des Containers mit der Trainingszeit des Sweeps und riss das
150-Sekunden-Zeitbudget des Keep-alive-Checks der Website. Der Sweep steht jetzt hinter einem
Button (`t_sweep_start`) – die Seite baut sich sofort auf, die Berechnung startet erst auf
Wunsch. Ergebnis serverseitig zwischengespeichert (`st.cache_data`), spätere Aufrufe sind sofort
fertig.

**Grenzen:** Nur ein Bit muss über die Zeit getragen werden (die einfachste Version der
Aufgabe) – komplexere Aufgaben (zwei Positionen in Beziehung setzen, wie Hochreiter &
Schmidhubers "Temporal Order") wären noch schwerer zu lernen. Die "Erfolg bricht zusammen"-Aussage
gilt für ein FESTES, realistisches Trainingsbudget, nicht als absolute Unmöglichkeit.

## Tests

30 Tests, `python -m pytest tests/ -v`:
- `test_scenario.py` – Reproduzierbarkeit, Signalposition, Klassenbalance.
- `test_model.py` – Forward/Backward, Gradienten-Check, Gradientennorm-Abfall,
  Korrektheits-Kette.
- `test_evaluation.py` – $T$-Sweep, Gradientennorm-Sweep.
- `test_presets.py`, `test_claims.py` – jede Zahl oben nachgerechnet.
- `test_app.py` – Streamlit `AppTest`: Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `rnn_constants.py` | Regler-Grenzen, Presets |
| `rnn_scenario.py` | Sequenzgenerator |
| `rnn_model.py` | RNN, SGD, Adam |
| `rnn_evaluation.py` | T-Sweep, Gradientennorm, Gradienten-Check, Reduktions-Check |
| `rnn_visualization.py` | Plotly-Plots |
| `rnn_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Keine komplexeren Sequenzaufgaben (zwei Positionen in Beziehung setzen) – das würde die klare
"ein Bit über die Zeit tragen"-Botschaft dieses Stücks verwässern. Kein Optimierer-Vergleich als
eigener Abschnitt (das wurde in Stück 2 abschließend behandelt) – Adam wird hier nur als
Plan-Korrektur erwähnt, nicht als eigene Messreihe wiederholt.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Elman, J. L. (1990). *Finding Structure in Time.* Cognitive Science, 14(2), 179–211.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8),
  1735–1780 (Adding Problem/Temporal Order als Vehikel-Vorlage).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html).
