"""Integración híbrida: un paso del RK4 existente + reseteo por evento."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .modelo import Neurona, derivadas


def cargar_rk4_repositorio():
    """Prioriza el RK4 instalado; de lo contrario localiza practicos/integradores/rk4."""
    try:
        from rk4lab.integradores import paso_rk4
        return paso_rk4
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        ruta = Path(__file__).resolve().parents[2] / "integradores" / "rk4" / "src"
        if not (ruta / "rk4lab" / "integradores.py").is_file():
            raise RuntimeError(
                "No se encontró rk4lab. Copiá tp1_izhikevich dentro de practicos/ "
                "(al lado de integradores/) o instalá rk4lab desde practicos/integradores/rk4."
            ) from None
        sys.path.insert(0, str(ruta))
        from rk4lab.integradores import paso_rk4
        return paso_rk4


@dataclass
class Traza:
    t: np.ndarray
    y: np.ndarray  # valores POST-RESET para continuar la dinámica
    v_grafico: np.ndarray  # +30 mV en los eventos de disparo
    I: np.ndarray
    spikes: np.ndarray
    parametros: Neurona


def integrar_rk4_con_reset(
    parametros: Neurona, corriente: Callable[[float], float],
    v0: float, tf: float, h: float = 0.1,
) -> Traza:
    paso_rk4 = cargar_rk4_repositorio()  # la implementación de rk4lab, NO una copia
    pasos = round(tf / h)
    if pasos <= 0 or not np.isclose(pasos * h, tf, atol=1e-10):
        raise ValueError("tf debe ser múltiplo entero de h y ambos positivos")
    t = np.linspace(0.0, tf, pasos + 1)
    y = np.empty((pasos + 1, 2))
    I = np.array([corriente(float(ti)) for ti in t], dtype=float)
    y[0] = [v0, parametros.b * v0]
    spikes = []
    v_grafico = np.empty(pasos + 1)
    v_grafico[0] = v0
    for k in range(pasos):
        # En cada subintervalo de corriente constante, RK4 ve I(t_k),
        # evitando mezclar ambos lados de una discontinuidad en k4.
        p = {"neurona": parametros, "corriente": I[k]}
        siguiente = paso_rk4(derivadas, float(t[k]), y[k], h, p)
        if siguiente[0] >= 30.0:
            spikes.append(t[k + 1])
            v_grafico[k + 1] = 30.0
            siguiente[0] = parametros.c
            siguiente[1] += parametros.d
        else:
            v_grafico[k + 1] = siguiente[0]
        y[k + 1] = siguiente
    return Traza(t, y, v_grafico, I, np.asarray(spikes), parametros)
