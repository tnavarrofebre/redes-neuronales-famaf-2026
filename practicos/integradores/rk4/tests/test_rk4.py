import numpy as np

from rk4lab.integradores import paso_rk4, resolver_rk4
from rk4lab.modelos import crecimiento_exponencial, decaimiento_exponencial, oscilador_armonico, raiz_no_unica
from rk4lab.analisis import estudio_orden_rk4, orden_observado


def test_decaimiento_exponencial():
    sol = resolver_rk4(decaimiento_exponencial, (0, 2), [1], 0.01, {"lambda": 1.0})
    assert abs(sol.y[-1, 0] - np.exp(-2)) < 1e-8


def test_oscilador_armonico_vectorial():
    sol = resolver_rk4(oscilador_armonico, (0, 2*np.pi), [1, 0], 0.005, {"omega": 1.0})
    assert sol.y.shape[1] == 2
    assert np.linalg.norm(sol.y[-1] - [1, 0]) < 1e-8


def test_un_paso_conocido_y_prima_igual_y():
    y1 = paso_rk4(crecimiento_exponencial, 0.0, np.array([1.0]), 1.0, {"lambda": 1.0})
    assert abs(y1[0] - 2.708333333333333) < 1e-14


def test_orden_local_y_global():
    pasos = 1 / (2.0 ** np.arange(2, 8))
    exacta = lambda t: np.array([np.exp(t)])
    _, e_local, e_global = estudio_orden_rk4(
        crecimiento_exponencial, exacta, [1.0], 3.0, pasos, {"lambda": 1.0}
    )
    # Usamos los ultimos cocientes, ya en regimen asintotico pero antes de redondeo dominante.
    assert abs(orden_observado(e_local)[-2] - 5.0) < 0.05
    assert abs(orden_observado(e_global)[-1] - 4.0) < 0.05


def test_problema_no_unico_permanece_en_cero():
    sol = resolver_rk4(raiz_no_unica, (0, 3), [0], 0.01, {})
    assert np.all(sol.y[:, 0] == 0.0)
