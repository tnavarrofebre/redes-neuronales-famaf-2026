#!/usr/bin/env python3
"""Ejecuta el TP1 completo, guarda datos y figuras (semilla fija)."""
from pathlib import Path
import json
import csv
import numpy as np

from src.modelo import protocolo
from src.simulacion import integrar_rk4_con_reset
from src.red import configurar_red, simular_red
from src.ruido import integrar_euler_maruyama
from src import figuras

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
DAT = BASE / "datos"


def guardar_traza(nombre, traza):
    np.savetxt(DAT / f"{nombre.lower()}_traza.csv",
               np.column_stack((traza.t, traza.v_grafico, traza.y[:, 0], traza.y[:, 1], traza.I)),
               delimiter=",", header="tiempo_ms,v_grafico_mV,v_post_reset_mV,u,I", comments="")
    np.savetxt(DAT / f"{nombre.lower()}_spikes.csv", traza.spikes, delimiter=",",
               header="tiempo_ms", comments="")


def guardar_spikes_red(nombre, red):
    datos = np.column_stack((red.tiempos_spike, red.indices_spike + 1))
    np.savetxt(DAT / f"{nombre}_spikes.csv", datos, delimiter=",", fmt="%d",
               header="tiempo_ms,neurona_indice_base_1", comments="")


def main():
    figuras.configurar()
    FIG.mkdir(exist_ok=True)
    DAT.mkdir(exist_ok=True)
    resumen = {"referencia": "E. M. Izhikevich (2003), DOI:10.1109/TNN.2003.820440",
               "paso_neurona_ms": 0.1, "red_semilla": 2026,
               "red_observacion_ms": 1000, "red_actualizacion": "dos Euler de 0.5 ms para v, un paso de 1 ms para u"}
    neurona, I, v0, tf, h = protocolo("BASE")
    base = integrar_rk4_con_reset(neurona, I, v0, tf, h)
    guardar_traza("BASE", base)
    figuras.grafico_base(base, FIG / "01_caso_base.png")
    resumen["caso_base"] = {"spikes": len(base.spikes), "tiempos_ms": base.spikes.round(3).tolist()}
    print(f"BASE: {len(base.spikes)} disparos: {np.round(base.spikes, 2)}")

    nombres = ("RS", "IB", "CH", "FS", "TC1", "TC2", "RZ", "LTS")
    casos = {}
    resumen["tipos"] = {}
    for nombre in nombres:
        neurona, I, v0, tf, h = protocolo(nombre)
        traza = integrar_rk4_con_reset(neurona, I, v0, tf, h)
        casos[nombre] = traza
        guardar_traza(nombre, traza)
        resumen["tipos"][nombre] = {"spikes": len(traza.spikes),
                                     "tiempos_ms": traza.spikes.round(3).tolist()}
        print(f"{nombre:>4}: {len(traza.spikes)} disparos")
    figuras.grafico_tipos(casos, FIG / "02_ocho_tipos.png")
    pasos = (0.2, 0.1, 0.05, 0.025)
    comparaciones = []
    n, I, v0, tf, h = protocolo("BASE")
    for hi in pasos:
        comparaciones.append((hi, integrar_rk4_con_reset(n, I, v0, tf, hi)))
    figuras.grafico_convergencia(comparaciones, FIG / "03_sensibilidad_paso.png")
    resumen["convergencia"] = [
        {"h_ms": hi, "primer_spike_ms": float(traza.spikes[0]) if len(traza.spikes) else None,
         "n_spikes": len(traza.spikes)} for hi, traza in comparaciones]

    config = configurar_red(n=1000, duracion_ms=1000, semilla=2026)
    con = simular_red(config, con_autoconexiones=True)
    sin = simular_red(config, con_autoconexiones=False)
    figuras.grafico_red(con, FIG / "04_red_original.png")
    figuras.grafico_comparacion_red(con, sin, FIG / "05_red_con_sin_autoconexiones.png")
    guardar_spikes_red("red_con_autoconexiones", con)
    guardar_spikes_red("red_sin_autoconexiones", sin)
    resumen["red"] = {}
    for clave, red in (("con_autoconexiones", con), ("sin_autoconexiones", sin)):
        resumen["red"][clave] = {"n": red.n, "ne": red.n_exc, "ni": red.n-red.n_exc,
                                 "n_spikes": red.total_spikes, "tasa_media_hz": round(red.tasa_hz, 4)}
        print(f"RED {clave:>18}: spikes={red.total_spikes}, tasa={red.tasa_hz:.3f} Hz")
    resumen["red"]["variacion_spikes_porcentual"] = round(
        100 * (sin.total_spikes - con.total_spikes) / con.total_spikes, 4)

    n, _, v0, *_ = protocolo("BASE")
    control = integrar_euler_maruyama(n, sigma=0, tf=200, h=0.01, semilla=2026)
    ruidosa = integrar_euler_maruyama(n, sigma=5, tf=200, h=0.01, semilla=2026)
    otra = integrar_euler_maruyama(n, sigma=5, tf=200, h=0.01, semilla=2027)
    figuras.grafico_ruido(control, ruidosa, otra, FIG / "06_una_neurona_ruido_euler_maruyama.png")
    for nombre, traza in (("euler_control", control), ("euler_ruido_2026", ruidosa),
                          ("euler_ruido_2027", otra)):
        np.savetxt(DAT / f"{nombre}.csv", np.column_stack((traza.t, traza.v, traza.u)),
                   delimiter=",", header="tiempo_ms,v_grafico_mV,u", comments="")
        np.savetxt(DAT / f"{nombre}_spikes.csv", traza.spikes,
                   delimiter=",", header="tiempo_ms", comments="")
    resumen["euler_maruyama"] = {
        "dt_ms": 0.01, "sigma_control": 0.0, "sigma_ruido": 5.0,
        "semillas": [2026, 2027], "n_spikes": {
            "control": len(control.spikes), "semilla_2026": len(ruidosa.spikes),
            "semilla_2027": len(otra.spikes)},
        "interpretacion_sigma": "coeficiente de ruido blanco continuo: sigma*dW con DeltaW=sqrt(dt)*N(0,1)"}
    print("EULER/EM spikes:", resumen["euler_maruyama"]["n_spikes"])
    (DAT / "resultados.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"Resultados escritos en {BASE}")


if __name__ == "__main__":
    main()
