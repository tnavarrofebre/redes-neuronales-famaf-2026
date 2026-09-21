"""Herramientas de analisis posteriores a la integracion."""
import numpy as np
from .integradores import paso_rk4, resolver_rk4


def seccion_poincare(solucion, periodo, t_inicio=0.0):
    """Muestrea la trayectoria en t_inicio+n*periodo mediante interpolacion lineal."""
    if periodo <= 0:
        raise ValueError("El periodo debe ser positivo.")
    tiempos = np.arange(t_inicio, solucion.t[-1] + 1e-12, periodo)
    tiempos = tiempos[(tiempos >= solucion.t[0]) & (tiempos <= solucion.t[-1])]
    estados = np.column_stack([
        np.interp(tiempos, solucion.t, solucion.y[:, j])
        for j in range(solucion.y.shape[1])
    ])
    return tiempos, estados


def estudio_orden_rk4(funcion, solucion_exacta, y0, tiempo_final, pasos,
                       parametros=None, componente=0):
    """Calcula errores local y global para una lista de pasos h.

    Error local: un unico paso h desde el dato inicial exacto.
    Error global: integracion completa hasta un T fijo.
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    pasos = np.asarray(pasos, dtype=float)
    errores_locales = []
    errores_globales = []

    for h in pasos:
        y_un_paso = paso_rk4(funcion, 0.0, y0, h, parametros)
        exacta_h = np.atleast_1d(solucion_exacta(h))
        errores_locales.append(abs(y_un_paso[componente] - exacta_h[componente]))

        solucion = resolver_rk4(funcion, (0.0, tiempo_final), y0, h, parametros)
        exacta_T = np.atleast_1d(solucion_exacta(tiempo_final))
        errores_globales.append(abs(solucion.y[-1, componente] - exacta_T[componente]))

    return pasos, np.asarray(errores_locales), np.asarray(errores_globales)


def orden_observado(errores):
    """p=log2(E(h)/E(h/2)); supone que los pasos se reducen por factor 2."""
    errores = np.asarray(errores, dtype=float)
    return np.log2(errores[:-1] / errores[1:])
