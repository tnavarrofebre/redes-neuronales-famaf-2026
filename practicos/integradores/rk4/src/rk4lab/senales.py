"""Senales externas reutilizables para forzar los modelos."""
import numpy as np


def constante(amplitud):
    return lambda t: float(amplitud)


def sinusoidal(amplitud, frecuencia, desplazamiento=0.0, fase=0.0):
    omega = 2 * np.pi * frecuencia
    return lambda t: desplazamiento + amplitud * np.sin(omega * t + fase)


def pulso(amplitud, t_inicio, t_fin, base=0.0):
    return lambda t: amplitud if t_inicio <= t <= t_fin else base


def sumar_senales(*senales):
    return lambda t: sum(senal(t) for senal in senales)
