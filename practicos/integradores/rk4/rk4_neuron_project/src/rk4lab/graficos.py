"""Funciones de visualizacion separadas del integrador numerico."""
import matplotlib.pyplot as plt


def graficar_temporal(solucion, indices=None, ax=None):
    ax = ax or plt.subplots()[1]
    indices = range(solucion.y.shape[1]) if indices is None else indices
    for i in indices:
        etiqueta = solucion.nombres[i] if solucion.nombres else f"y[{i}]"
        ax.plot(solucion.t, solucion.y[:, i], label=etiqueta)
    ax.set(xlabel="t", ylabel="estado")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def graficar_fase(solucion, i=0, j=1, ax=None):
    ax = ax or plt.subplots()[1]
    ax.plot(solucion.y[:, i], solucion.y[:, j])
    xi = solucion.nombres[i] if solucion.nombres else f"y[{i}]"
    xj = solucion.nombres[j] if solucion.nombres else f"y[{j}]"
    ax.set(xlabel=xi, ylabel=xj)
    ax.grid(True, alpha=0.25)
    return ax


def graficar_convergencia(pasos, errores_locales, errores_globales, ax=None):
    """Grafico log-log de los errores y referencias h^5 y h^4."""
    ax = ax or plt.subplots()[1]
    ax.loglog(pasos, errores_locales, "o-", label="Error local")
    ax.loglog(pasos, errores_globales, "s-", label="Error global")

    # Referencias escaladas al ultimo punto para comparar pendientes, no magnitudes.
    ref5 = errores_locales[-1] * (pasos / pasos[-1])**5
    ref4 = errores_globales[-1] * (pasos / pasos[-1])**4
    ax.loglog(pasos, ref5, "--", label=r"Referencia $h^5$")
    ax.loglog(pasos, ref4, "--", label=r"Referencia $h^4$")
    ax.set(xlabel="Paso h", ylabel="Error absoluto")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return ax
