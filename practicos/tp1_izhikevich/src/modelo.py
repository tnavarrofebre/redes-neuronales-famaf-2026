"""Modelo de Izhikevich y protocolos especificados en el TP1 (FAMAF, 2026)."""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Neurona:
    a: float = 0.02
    b: float = 0.20
    c: float = -65.0
    d: float = 2.0


def derivadas(t: float, y: np.ndarray, parametros: dict) -> np.ndarray:
    """F(t, Y, p) para rk4lab; I congelada en el paso que contiene un salto."""
    v, u = y
    n = parametros["neurona"]
    corriente = parametros["corriente"]
    return np.array([0.04 * v * v + 5.0 * v + 140.0 - u + corriente,
                     n.a * (n.b * v - u)], dtype=float)


def escalon(t: float, instante: float, amplitud: float) -> float:
    return float(amplitud) if t >= instante - 1e-10 else 0.0


def protocolo(nombre: str) -> tuple[Neurona, Callable[[float], float], float, float, float]:
    """Devuelve parámetros, I(t), v0, tf y h de acuerdo con el TP1."""
    if nombre == "BASE":
        return Neurona(0.02, 0.20, -65, 2), lambda t: escalon(t, 10, 10), -70, 100, 0.1
    parametros = {
        "RS": Neurona(0.02, 0.20, -65, 8),
        "IB": Neurona(0.02, 0.20, -55, 4),
        "CH": Neurona(0.02, 0.20, -50, 2),
        "FS": Neurona(0.10, 0.20, -65, 2),
        "TC1": Neurona(0.02, 0.25, -65, 0.05),
        "TC2": Neurona(0.02, 0.25, -65, 0.05),
        "RZ": Neurona(0.10, 0.26, -65, 2),
        "LTS": Neurona(0.02, 0.25, -65, 2),
    }
    if nombre not in parametros:
        raise ValueError(f"Caso desconocido: {nombre}")
    n = parametros[nombre]
    if nombre == "TC1":
        return n, lambda t: escalon(t, 20, 1), -63, 200, 0.1
    if nombre == "TC2":
        def corriente(t):
            return -25.0 if 20 - 1e-10 <= t < 80 - 1e-10 else 0.0
        return n, corriente, -65, 200, 0.1
    if nombre == "RZ":
        def corriente(t):
            if t < 40 - 1e-10:
                return 0.0
            if t < 180 - 1e-10:
                return 0.2
            if t < 181 - 1e-10:
                return 2.2
            return 0.2
        return n, corriente, -62.5, 320, 0.1
    return n, lambda t: escalon(t, 20, 10), -65, 200, 0.1
