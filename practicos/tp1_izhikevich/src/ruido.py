"""Euler y Euler-Maruyama locales al TP1: NO se modifica rk4lab."""
from dataclasses import dataclass
import numpy as np
from .modelo import Neurona, escalon


@dataclass
class ResultadoRuido:
    t: np.ndarray
    v: np.ndarray  # valores con pico visible de 30mV
    u: np.ndarray
    spikes: np.ndarray
    sigma: float
    semilla: int


def integrar_euler_maruyama(
    neurona: Neurona, *, h: float = 0.01, tf: float = 200.0,
    v0: float = -70.0, sigma: float = 5.0, semilla: int = 2026,
    instante_estimulo: float = 10.0, amplitud_estimulo: float = 10.0,
) -> ResultadoRuido:
    """dV=[0.04V²+5V+140-U+I(t)]dt+sigma*dW, dU=a(bV-U)dt.

    Euler-Maruyama: DeltaW=sqrt(h)*N(0,1), generado independientemente
    por paso. El reseteo es un evento DISCRETO, después de cada paso.
    sigma=0 es Euler explícito determinista.
    """
    if h <= 0 or sigma < 0:
        raise ValueError("h debe ser positivo y sigma no negativo")
    pasos = round(tf / h)
    if not np.isclose(pasos * h, tf):
        raise ValueError("tf debe ser múltiplo de h")
    t = np.linspace(0, tf, pasos + 1)
    v = np.empty(pasos + 1)
    u = np.empty(pasos + 1)
    v[0], u[0] = v0, neurona.b * v0
    rng = np.random.default_rng(semilla)
    xi = rng.standard_normal(pasos)
    spikes = []
    # Estado real post reset, distinto al pico mostrado en v.
    estado_v = float(v0)
    estado_u = float(neurona.b * v0)
    for k in range(pasos):
        I = escalon(float(t[k]), instante_estimulo, amplitud_estimulo)
        dv = 0.04 * estado_v**2 + 5 * estado_v + 140 - estado_u + I
        du = neurona.a * (neurona.b * estado_v - estado_u)
        nuevo_v = estado_v + h * dv + sigma * np.sqrt(h) * xi[k]
        nuevo_u = estado_u + h * du
        if nuevo_v >= 30.0:
            spikes.append(t[k + 1])
            v[k + 1] = 30.0
            estado_v = neurona.c
            estado_u = nuevo_u + neurona.d
        else:
            estado_v, estado_u = nuevo_v, nuevo_u
            v[k + 1] = estado_v
        u[k + 1] = estado_u
    return ResultadoRuido(t, v, u, np.asarray(spikes), sigma, semilla)
