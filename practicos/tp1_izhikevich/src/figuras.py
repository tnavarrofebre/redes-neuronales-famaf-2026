"""Figuras legibles para informe y para consultar individualmente."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL = "#2166ac"
ROJO = "#b2182b"
VERDE = "#1b7837"
GRIS = "#767676"


def configurar():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 130, "savefig.dpi": 200,
        "axes.grid": True, "grid.alpha": 0.18,
        "savefig.bbox": "tight", "lines.linewidth": 1.15,
    })


def guardar(fig, ruta):
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)


def grafico_base(resultado, ruta):
    fig, ax = plt.subplots(3, 1, figsize=(11.0, 6.3), sharex=True,
                           gridspec_kw={"height_ratios": [2.3, 1.0, 0.75]})
    ax[0].plot(resultado.t, resultado.v_grafico, color=AZUL)
    if resultado.spikes.size:
        ax[0].scatter(resultado.spikes, np.full_like(resultado.spikes, 30),
                      s=13, color=ROJO, zorder=4, label="Disparos (+30 mV)")
        ax[0].legend(loc="upper right", frameon=False, fontsize=9)
    ax[0].set(ylabel="v (mV)", title="Parte I: potencial y recuperación — caso base (RK4, h = 0,1 ms)")
    ax[1].plot(resultado.t, resultado.y[:, 1], color=VERDE)
    ax[1].set(ylabel="u")
    ax[2].step(resultado.t, resultado.I, where="post", color=GRIS)
    ax[2].set(ylabel="I", xlabel="Tiempo (ms)", ylim=(-0.8, 12))
    for a in ax:
        a.axvline(10, linestyle="--", color=ROJO, lw=1.0, alpha=0.65)
    fig.tight_layout()
    guardar(fig, ruta)


def grafico_tipos(casos, ruta):
    titulos = {
        "RS": "RS · disparo regular", "IB": "IB · ráfaga inicial",
        "CH": "CH · chattering", "FS": "FS · disparo rápido",
        "TC1": "TC1 · tónico", "TC2": "TC2 · rebote",
        "RZ": "RZ · resonancia", "LTS": "LTS · umbral bajo",
    }
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 10.4))
    for a, (nombre, traza) in zip(axes.flat, casos.items()):
        a.plot(traza.t, traza.v_grafico, color=AZUL, lw=0.95)
        a.set(ylabel="v (mV)", xlabel="t (ms)", title=f"{titulos[nombre]} ({len(traza.spikes)} spikes)")
        a.set_ylim(-105 if nombre == "TC2" else -90, 40)
        a2 = a.twinx()
        a2.grid(False)
        a2.plot(traza.t, traza.I, color=ROJO, alpha=0.38, lw=1, drawstyle="steps-post")
        a2.set_ylabel("I", color=ROJO, fontsize=8)
        a2.tick_params(axis="y", colors=ROJO, labelsize=7)
        a2.spines["right"].set_visible(True)
        a2.spines["top"].set_visible(False)
    fig.suptitle("Ocho patrones de Izhikevich · protocolos del TP1 · RK4 (h = 0,1 ms)", y=1.007, fontsize=13)
    fig.tight_layout()
    guardar(fig, ruta)


def grafico_red(red, ruta):
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [3.6, 1.05, 1.15]})
    es_exc = red.indices_spike < red.n_exc
    axs[0].scatter(red.tiempos_spike[es_exc], red.indices_spike[es_exc] + 1,
                   s=0.55, color=AZUL, rasterized=True, label="Excitatorias")
    axs[0].scatter(red.tiempos_spike[~es_exc], red.indices_spike[~es_exc] + 1,
                   s=0.55, color=ROJO, rasterized=True, label="Inhibitorias")
    axs[0].axhline(red.n_exc + 0.5, color=GRIS, ls="--", lw=0.9)
    axs[0].set(ylabel="Índice neuronal", title=f"Red original ({red.n} neuronas) · Raster de disparos · {red.tasa_hz:.2f} Hz promedio")
    axs[0].legend(loc="upper right", markerscale=6, frameon=False, ncol=2)
    axs[1].plot(np.arange(len(red.voltaje_ejemplo)), red.voltaje_ejemplo, color=AZUL, lw=0.8)
    axs[1].set(ylabel="v₁ (mV)", ylim=(-95, 40))
    duracion = red.duracion_ms
    bordes = np.arange(0, duracion + 10, 10)
    conteos, _ = np.histogram(red.tiempos_spike, bins=bordes)
    poblacion_hz = conteos / (red.n * 0.010)
    axs[2].step(bordes[:-1], poblacion_hz, where="post", color=VERDE)
    axs[2].set(xlabel="Tiempo (ms)", ylabel="Tasa (Hz/ne.)", xlim=(0, duracion))
    fig.tight_layout()
    guardar(fig, ruta)


def grafico_comparacion_red(con, sin, ruta):
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [2, 2, 1.1]})
    for ax, red, titulo in zip(axs[:2], [con, sin], ["Con autoconexiones (original)", "Sin autoconexiones (diag(S) = 0)"]):
        ax.scatter(red.tiempos_spike, red.indices_spike + 1,
                   s=0.48, color=AZUL if red.con_autoconexiones else ROJO, rasterized=True)
        ax.set(ylabel="Neurona", title=f"{titulo} — {red.total_spikes} spikes; {red.tasa_hz:.2f} Hz")
        ax.axhline(red.n_exc + 0.5, color=GRIS, lw=0.75, ls="--")
    bins = np.arange(0, con.duracion_ms + 20, 20)
    for red, col, nom in [(con, AZUL, "Con diagonal"), (sin, ROJO, "Sin diagonal")]:
        ct, _ = np.histogram(red.tiempos_spike, bins=bins)
        axs[2].step(bins[:-1], ct / (red.n * 0.02), where="post", label=nom, color=col)
    axs[2].set(xlabel="Tiempo (ms)", ylabel="Tasa (Hz/ne.)", xlim=(0, con.duracion_ms))
    axs[2].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    guardar(fig, ruta)


def grafico_ruido(determinista, ruidosa, otra, ruta):
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1]})
    for a, res, color, etiqueta in [
        (axs[0], determinista, GRIS, "Euler, σ = 0 (control)"),
        (axs[1], ruidosa, AZUL, f"Euler–Maruyama, σ = {ruidosa.sigma:g}, semilla {ruidosa.semilla}"),
        (axs[2], otra, ROJO, f"Euler–Maruyama, σ = {otra.sigma:g}, semilla {otra.semilla}"),
    ]:
        a.plot(res.t, res.v, lw=0.75, color=color)
        a.set(ylabel="v (mV)", ylim=(-90, 40), title=f"{etiqueta} · {len(res.spikes)} spikes")
        a.axvline(10, color=GRIS, ls="--", alpha=0.55)
    axs[2].set_xlabel("Tiempo (ms)")
    axs[2].set_xlim(0, len(ruidosa.t) * (ruidosa.t[1] - ruidosa.t[0]) - (ruidosa.t[1] - ruidosa.t[0]))
    fig.suptitle("Una neurona con ruido blanco: comparación de realizaciones", y=1.006)
    fig.tight_layout()
    guardar(fig, ruta)


def grafico_convergencia(trazas, ruta):
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.4))
    colores = plt.get_cmap("viridis")
    for i, (h, traza) in enumerate(trazas):
        mask = (traza.t >= 10) & (traza.t <= 38)
        axs[0].plot(traza.t[mask], traza.v_grafico[mask], label=f"h={h:g} ms",
                    lw=1.05, alpha=0.83, color=colores(i / max(1, len(trazas)-1)))
    axs[0].set(xlabel="Tiempo (ms)", ylabel="v (mV)", title="Convergencia de trayectorias (ventana inicial)")
    axs[0].legend(frameon=False, fontsize=8)
    hs = np.array([h for h, traza in trazas])
    primeros = np.array([traza.spikes[0] if len(traza.spikes) else np.nan for h, traza in trazas])
    axs[1].plot(hs, primeros, "o-", color=AZUL)
    axs[1].set(xlabel="Paso h (ms)", ylabel="Tiempo del primer spike (ms)", title="Sensibilidad temporal del primer disparo")
    axs[1].invert_xaxis()
    fig.tight_layout()
    guardar(fig, ruta)
