"""Ejecuta ejemplos de validacion y genera las figuras del proyecto."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rk4lab.integradores import resolver_rk4
from rk4lab.modelos import (
    decaimiento_exponencial,
    oscilador_armonico,
    sistema_lineal_dos_entradas,
    duffing,
    lorenz,
    raiz_no_unica,
)
from rk4lab.senales import sinusoidal, pulso
from rk4lab.graficos import graficar_temporal, graficar_fase, graficar_convergencia
from rk4lab.analisis import seccion_poincare, estudio_orden_rk4, orden_observado
from rk4lab.modelos import crecimiento_exponencial

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"
FIGURAS.mkdir(exist_ok=True)

# 1) EDO escalar con solucion exacta y=e^-t
s1 = resolver_rk4(decaimiento_exponencial, (0, 5), [1], 0.01, {"lambda": 1}, nombres=["y"])
ax = graficar_temporal(s1)
ax.plot(s1.t, np.exp(-s1.t), "--", label="solucion exacta")
ax.legend(); ax.set_title("1. Decaimiento exponencial")
ax.figure.savefig(FIGURAS / "01_decaimiento_exponencial.png", dpi=160, bbox_inches="tight")
plt.close(ax.figure)

# 2) EDO de segundo orden convertida a sistema: Y=(x,v)
s2 = resolver_rk4(oscilador_armonico, (0, 20), [1, 0], 0.01, {"omega": 1}, nombres=["x", "v"])
ax = graficar_fase(s2)
ax.set_title("2. Oscilador armonico: espacio de fase")
ax.figure.savefig(FIGURAS / "02_oscilador_fase.png", dpi=160, bbox_inches="tight")
plt.close(ax.figure)

# 3) Una variable de estado con DOS entradas simultaneas
p3 = {
    "a": 1.2, "b1": 1.0, "b2": 0.6,
    "entrada1": sinusoidal(1, 0.5),
    "entrada2": pulso(2, 3, 6),
}
s3 = resolver_rk4(sistema_lineal_dos_entradas, (0, 12), [0], 0.005, p3, nombres=["salida"])
fig, ax = plt.subplots()
ax.plot(s3.t, [p3["entrada1"](t) for t in s3.t], label="entrada 1")
ax.plot(s3.t, [p3["entrada2"](t) for t in s3.t], label="entrada 2")
ax.plot(s3.t, s3.y[:, 0], label="salida")
ax.set(xlabel="t", ylabel="amplitud", title="3. Dos entradas simultaneas y una salida")
ax.grid(True, alpha=.25); ax.legend()
fig.savefig(FIGURAS / "03_dos_entradas.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 4) Duffing: no lineal, forzado y seccion de Poincare
frecuencia = 0.2
p4 = {"delta": 0.2, "alpha": -1, "beta": 1, "entrada": sinusoidal(0.3, frecuencia)}
s4 = resolver_rk4(duffing, (0, 300), [0.1, 0], 0.02, p4, nombres=["x", "v"])
_, puntos = seccion_poincare(s4, 1 / frecuencia, t_inicio=100)
fig, ax = plt.subplots()
ax.scatter(puntos[:, 0], puntos[:, 1], s=10)
ax.set(xlabel="x", ylabel="v", title="4. Duffing: seccion de Poincare")
ax.grid(True, alpha=.25)
fig.savefig(FIGURAS / "04_duffing_poincare.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 5) Lorenz: tres estados no lineales acoplados
s5 = resolver_rk4(lorenz, (0, 30), [1, 1, 1], 0.005, {}, nombres=["x", "y", "z"])
ax = graficar_fase(s5, 0, 2)
ax.set_title("5. Lorenz: proyeccion x-z")
ax.figure.savefig(FIGURAS / "05_lorenz.png", dpi=160, bbox_inches="tight"); plt.close(ax.figure)

# 6) Verificacion experimental del orden: y'=y, y(0)=1, y=e^t
pasos = 1 / (2.0 ** np.arange(0, 8))
exacta = lambda t: np.array([np.exp(t)])
hs, e_local, e_global = estudio_orden_rk4(
    crecimiento_exponencial, exacta, [1.0], 3.0, pasos, {"lambda": 1.0}
)
ax = graficar_convergencia(hs, e_local, e_global)
ax.set_title("6. Convergencia de RK4")
ax.figure.savefig(FIGURAS / "06_convergencia_rk4.png", dpi=160, bbox_inches="tight"); plt.close(ax.figure)

print("\nEstudio de convergencia para y'=y, y(0)=1, T=3")
print(" h          error local       p_local    error global      p_global")
p_local = orden_observado(e_local)
p_global = orden_observado(e_global)
for i, h in enumerate(hs):
    pl = "-" if i == 0 else f"{p_local[i-1]:.5f}"
    pg = "-" if i == 0 else f"{p_global[i-1]:.5f}"
    print(f"{h:<10.7f} {e_local[i]:<17.9e} {pl:<10} {e_global[i]:<17.9e} {pg}")

# 7) Caso patologico: y'=2 sqrt(y), y(0)=0. RK4 sigue y=0, tambien solucion exacta.
s7 = resolver_rk4(raiz_no_unica, (0, 3), [0], 0.01, {})
assert np.allclose(s7.y[:, 0], 0.0)
print("\nCaso no unico y'=2*sqrt(y), y(0)=0: RK4 permanece en y=0, como corresponde a una solucion valida.")
print(f"Figuras guardadas en: {FIGURAS}")
