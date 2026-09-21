"""Modelos de prueba escritos en la forma vectorial Y' = F(t,Y,parametros)."""
import numpy as np


def crecimiento_exponencial(t, y, p):
    """y' = lambda*y + entrada(t). Con lambda=1: y=e^t si y(0)=1."""
    entrada = p.get("entrada", lambda _t: 0.0)
    return np.array([p.get("lambda", 1.0) * y[0] + entrada(t)])


def decaimiento_exponencial(t, y, p):
    """y' = -lambda*y + entrada(t)."""
    entrada = p.get("entrada", lambda _t: 0.0)
    return np.array([-p.get("lambda", 1.0) * y[0] + entrada(t)])


def oscilador_armonico(t, y, p):
    """x'' + omega^2*x = entrada(t), transformada a Y=(x,v)."""
    x, v = y
    omega = p.get("omega", 1.0)
    fuerza = p.get("entrada", lambda _t: 0.0)(t)
    return np.array([v, -(omega**2) * x + fuerza])


def sistema_lineal_dos_entradas(t, y, p):
    """y' = -a*y + b1*u1(t) + b2*u2(t)."""
    u1 = p["entrada1"](t)
    u2 = p["entrada2"](t)
    return np.array([
        -p.get("a", 1.0) * y[0]
        + p.get("b1", 1.0) * u1
        + p.get("b2", 1.0) * u2
    ])


def duffing(t, y, p):
    """Oscilador de Duffing: x''+delta*x'+alpha*x+beta*x^3=entrada(t)."""
    x, v = y
    fuerza = p.get("entrada", lambda _t: 0.0)(t)
    return np.array([
        v,
        -p["delta"] * v - p["alpha"] * x - p["beta"] * x**3 + fuerza,
    ])


def lorenz(t, y, p):
    """Sistema no lineal tridimensional de Lorenz."""
    x, yy, z = y
    sigma = p.get("sigma", 10.0)
    rho = p.get("rho", 28.0)
    beta = p.get("beta", 8 / 3)
    return np.array([
        sigma * (yy - x),
        x * (rho - z) - yy,
        x * yy - beta * z,
    ])


def neurona_lif_subumbral(t, y, p):
    """Dinamica subumbral LIF; el disparo/reset es un evento separado."""
    V = y[0]
    corriente = p.get("entrada", lambda _t: 0.0)(t)
    return np.array([(-(V - p["E_L"]) + p["R_m"] * corriente) / p["tau_m"]])


def raiz_no_unica(t, y, p=None):
    """y'=2*sqrt(y), y(0)=0: ejemplo de falta de unicidad."""
    return np.array([2 * np.sqrt(max(y[0], 0.0))])
