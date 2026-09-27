"""Klassisches Elman-RNN mit Backpropagation Through Time (BPTT) von Hand,
Perceptron-Criterion-Verlust (dieselbe Familie wie Stück 1/2) und Adam.

Ausgabe nur am letzten Zeitschritt: s = Why . h_T + by. Bei Fehlklassifikation
(y*s <= 0) wird der Fehler rekursiv durch die Zeit zurueckgereicht:
delta_T = (Why * (-y)) o (1-h_T^2), delta_t = (Whh^T . delta_{t+1}) o (1-h_t^2)
fuer t=T-1..1 - die Norm von delta_1 gegen delta_T ist die direkte Messung
des verschwindenden Gradienten (siehe rnn_evaluation.gradient_norms)."""
from dataclasses import dataclass, field

import numpy as np


class RNN:
    def __init__(self, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = n_hidden
        limit_xh = np.sqrt(6.0 / (1 + H))
        limit_hh = np.sqrt(6.0 / (H + H))
        limit_hy = np.sqrt(6.0 / (H + 1))
        self.H = H
        self.params = {
            "Wxh": rng.uniform(-limit_xh, limit_xh, size=(H, 1)),
            "Whh": rng.uniform(-limit_hh, limit_hh, size=(H, H)),
            "bh": np.zeros(H),
            "Why": rng.uniform(-limit_hy, limit_hy, size=H),
            "by": np.zeros(()),
        }

    def forward(self, seq: np.ndarray):
        T = len(seq)
        h = np.zeros(self.H)
        hs = [h]
        p = self.params
        for t in range(T):
            z = p["Wxh"][:, 0] * seq[t] + p["Whh"] @ h + p["bh"]
            h = np.tanh(z)
            hs.append(h)
        s = float(p["Why"] @ hs[-1] + p["by"])
        return s, hs

    def backward(self, y: float, s: float, hs: list, seq: np.ndarray):
        """Gibt (grads, deltas) zurueck; deltas[0] = Gradient am ERSTEN
        Zeitschritt, deltas[-1] = Gradient am LETZTEN Zeitschritt. Bei
        korrekter Klassifikation (y*s>0): grads=Nullen, deltas=None."""
        T = len(seq)
        p = self.params
        grads = {"Wxh": np.zeros_like(p["Wxh"]), "Whh": np.zeros_like(p["Whh"]),
                 "bh": np.zeros_like(p["bh"]), "Why": np.zeros_like(p["Why"]),
                 "by": np.zeros_like(p["by"])}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * hs[-1]
        grads["by"] = np.asarray(ds)
        dh_next = ds * p["Why"]
        deltas = []
        for t in range(T, 0, -1):
            delta = dh_next * (1 - hs[t] ** 2)
            deltas.append(delta)
            grads["Wxh"][:, 0] += delta * seq[t - 1]
            grads["Whh"] += np.outer(delta, hs[t - 1])
            grads["bh"] += delta
            dh_next = p["Whh"].T @ delta
        deltas.reverse()
        return grads, deltas

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0

    def n_params(self) -> int:
        return sum(v.size for v in self.params.values())


class SGD:
    """Einfacher Gradientenschritt (keine adaptive Normalisierung). Bewusst
    NICHT Adam als Standard hier: Adams pro-Parameter-Normalisierung (Division
    durch die eigene historische Gradientengroesse) kompensiert einen Teil des
    verschwindenden Gradienten und verwaesserte in der Vormessung genau den
    Effekt, den dieses Stueck zeigen soll (siehe README, Plan-Korrektur)."""
    def __init__(self, eta: float = 0.1):
        self.eta = eta

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            params[key] -= self.eta * g


class Adam:
    def __init__(self, eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for key, g in grads.items():
            m = self.beta1 * self.m.get(key, np.zeros_like(g)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get(key, np.zeros_like(g)) + (1 - self.beta2) * g ** 2
            self.m[key] = m
            self.v[key] = v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[key] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class TrainResult:
    errors_per_epoch: list = field(default_factory=list)


def train_epoch(rnn: RNN, optimizer: Adam, X: np.ndarray, y: np.ndarray) -> int:
    n_errors = 0
    for seq, label in zip(X, y):
        s, hs = rnn.forward(seq)
        grads, deltas = rnn.backward(label, s, hs, seq)
        if deltas is not None:
            n_errors += 1
            optimizer.step(rnn.params, grads)
    return n_errors


def train(rnn: RNN, optimizer: Adam, X: np.ndarray, y: np.ndarray, epochs: int) -> TrainResult:
    errors_per_epoch = []
    for _ in range(epochs):
        errors_per_epoch.append(train_epoch(rnn, optimizer, X, y))
    return TrainResult(errors_per_epoch)


def accuracy(rnn: RNN, X: np.ndarray, y: np.ndarray) -> float:
    correct = sum(rnn.predict(seq) == label for seq, label in zip(X, y))
    return correct / len(y)
