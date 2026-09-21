"""Integradores numericos para sistemas de EDO de primer orden."""
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

Arreglo = np.ndarray
FuncionEDO = Callable[[float, Arreglo, Any], Arreglo]


@dataclass
class Solucion:
    """Resultado de una integracion temporal."""
    t: Arreglo
    y: Arreglo  # forma: (numero_de_tiempos, numero_de_estados)
    parametros: Any = None
    nombres: tuple[str, ...] | None = None

    def estado(self, i: int) -> Arreglo:
        """Devuelve la evolucion temporal de la componente i del estado."""
        return self.y[:, i]


def paso_rk4(funcion: FuncionEDO, t: float, y: Arreglo, h: float, parametros=None) -> Arreglo:
    """Realiza un paso del Runge-Kutta clasico de cuarto orden.

    Resuelve localmente y' = F(t, y). Es un metodo explicito: k1, k2, k3 y k4
    se evaluan sucesivamente usando solo cantidades ya conocidas; no se resuelve
    ningun sistema algebraico.
    """
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(funcion(t, y, parametros), dtype=float)
    k2 = np.asarray(funcion(t + h / 2, y + h * k1 / 2, parametros), dtype=float)
    k3 = np.asarray(funcion(t + h / 2, y + h * k2 / 2, parametros), dtype=float)
    k4 = np.asarray(funcion(t + h, y + h * k3, parametros), dtype=float)

    if not (k1.shape == k2.shape == k3.shape == k4.shape == y.shape):
        raise ValueError("F(t, y, parametros) debe devolver un vector con la misma forma que y.")

    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def resolver_rk4(funcion: FuncionEDO, intervalo_t, y0, dt: float,
                  parametros=None, nombres=None) -> Solucion:
    """Integra y' = F(t,y) con RK4 clasico y paso maximo fijo dt.

    Si el ultimo paso no entra exactamente en el intervalo, se acorta para llegar
    exactamente a tf.
    """
    t0, tf = map(float, intervalo_t)
    if dt <= 0:
        raise ValueError("dt debe ser positivo.")
    if tf <= t0:
        raise ValueError("El tiempo final debe ser mayor que el inicial.")

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    numero_pasos = int(np.ceil((tf - t0) / dt))
    t = np.empty(numero_pasos + 1)
    y = np.empty((numero_pasos + 1, y0.size))
    t[0], y[0] = t0, y0

    for n in range(numero_pasos):
        h = min(dt, tf - t[n])
        t[n + 1] = t[n] + h
        y[n + 1] = paso_rk4(funcion, t[n], y[n], h, parametros)

    return Solucion(
        t=t,
        y=y,
        parametros=parametros,
        nombres=tuple(nombres) if nombres else None,
    )
