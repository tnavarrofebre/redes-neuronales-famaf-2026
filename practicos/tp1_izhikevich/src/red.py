"""Reproducción del algoritmo de la Sección IV del artículo de Izhikevich (2003)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ConfigRed:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    S: np.ndarray
    ruido: np.ndarray
    n_excitatorias: int
    semilla: int


@dataclass
class ResultadoRed:
    tiempos_spike: np.ndarray
    indices_spike: np.ndarray
    voltaje_ejemplo: np.ndarray
    n: int
    n_exc: int
    duracion_ms: int
    con_autoconexiones: bool

    @property
    def total_spikes(self) -> int:
        return int(len(self.tiempos_spike))

    @property
    def tasa_hz(self) -> float:
        return self.total_spikes / (self.n * (self.duracion_ms / 1000.0))


def configurar_red(n: int = 1000, duracion_ms: int = 1000, semilla: int = 2026) -> ConfigRed:
    """N divisible por 5; pesos y ruido se muestrean una sola vez para comparación pareada."""
    if n % 5 != 0 or n < 5:
        raise ValueError("N debe ser divisible por cinco y >= 5")
    ne, ni = 4 * n // 5, n // 5
    rng = np.random.default_rng(semilla)
    re, ri = rng.random(ne), rng.random(ni)
    a = np.r_[np.full(ne, 0.02), 0.02 + 0.08 * ri]
    b = np.r_[np.full(ne, 0.20), 0.25 - 0.05 * ri]
    c = np.r_[-65 + 15 * re**2, np.full(ni, -65.0)]
    d = np.r_[8 - 6 * re**2, np.full(ni, 2.0)]
    S = np.empty((n, n), dtype=float)
    S[:, :ne] = 0.5 * rng.random((n, ne))
    S[:, ne:] = -rng.random((n, ni))
    amplitudes = np.r_[np.full(ne, 5.0), np.full(ni, 2.0)]
    # Cada columna de ruido es una unidad de tiempo de 1 ms; reutilizada
    # exactamente en las dos redes. Distribución gaussiana por paso (NO SDE).
    ruido = rng.standard_normal((duracion_ms, n)) * amplitudes[None, :]
    return ConfigRed(a, b, c, d, S, ruido, ne, semilla)


def simular_red(config: ConfigRed, con_autoconexiones: bool = True) -> ResultadoRed:
    """Algoritmo original: disparos al INICIO de cada ms, 2 Euler de 0.5 ms para v."""
    a, b, c, d = config.a, config.b, config.c, config.d
    n = a.size
    # No se modifica S original: así se comparte entre ejecuciones.
    S = config.S if con_autoconexiones else config.S.copy()
    if not con_autoconexiones:
        np.fill_diagonal(S, 0.0)
    v = np.full(n, -65.0)
    u = b * v
    tiempos, indices = [], []
    v_ejemplo = np.empty(len(config.ruido) + 1)
    v_ejemplo[0] = v[0]
    for k, I_externo in enumerate(config.ruido):
        dispararon = np.flatnonzero(v >= 30.0)
        if dispararon.size:
            tiempos.extend([k + 1] * len(dispararon))
            indices.extend(dispararon.tolist())
            v[dispararon] = c[dispararon]
            u[dispararon] += d[dispararon]
        I_total = I_externo.copy()
        if dispararon.size:
            I_total += S[:, dispararon].sum(axis=1)
        # Igual que el Matlab original: u permanece congelada durante los
        # dos medios pasos de v; después se actualiza una vez con h=1 ms.
        v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I_total)
        v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I_total)
        u += a * (b * v - u)
        v_ejemplo[k + 1] = 30.0 if 0 in dispararon else v[0]
    return ResultadoRed(np.array(tiempos, dtype=int), np.array(indices, dtype=int),
                        v_ejemplo, n, config.n_excitatorias, len(config.ruido),
                        con_autoconexiones)
