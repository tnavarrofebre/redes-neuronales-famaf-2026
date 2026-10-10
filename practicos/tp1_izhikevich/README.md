# TP1 — Modelo de neuronas de Izhikevich

**Fundamentos de Redes Neuronales, FAMAF 2026.** Resolución de `tp1-2026.pdf`, basada en E. M. Izhikevich, *Simple Model of Spiking Neurons*, IEEE Trans. Neural Networks **14**(6), 1569–1572 (2003), DOI: [10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440).

Este directorio está pensado para copiarse en `practicos/tp1_izhikevich/`, **junto a** `practicos/integradores/rk4/`. La integración determinista de cada neurona emplea **`paso_rk4` del paquete `rk4lab` existente**, sin duplicar su implementación. Los reinicios se aplican tras cada paso. La red usa expresamente el algoritmo de dos medios pasos de Euler del artículo. La neurona estocástica utiliza un integrador Euler–Maruyama definido *en este mismo TP*.

## Ejecutar

Desde la raíz del repositorio, tras instalar las dependencias:

```bash
python -m pip install -r practicos/tp1_izhikevich/requirements.txt
python practicos/tp1_izhikevich/ejecutar_tp1.py
```

Si ejecutás fuera del repositorio, instalá antes el paquete `rk4lab` desde `practicos/integradores/rk4` mediante `pip install -e .`.

Para pruebas desde `practicos/tp1_izhikevich/`:

```bash
python -m pytest -q
```

## Resultados ya incluidos

- `figuras/01_caso_base.png`: `v(t)`, `u(t)`, `I(t)` y tiempos de disparo.
- `figuras/02_ocho_tipos.png`: RS, IB, CH, FS, TC1, TC2, RZ y LTS de la Fig. 2 del artículo.
- `figuras/03_sensibilidad_paso.png`: efecto de `h` en la dinámica y la fecha del primer disparo.
- `figuras/04_red_original.png`: Fig. 3 del artículo, aprox. (1000 neuronas).
- `figuras/05_red_con_sin_autoconexiones.png`: dos redes con idénticos parámetros, ruido y pesos fuera de la diagonal.
- `figuras/06_una_neurona_ruido_euler_maruyama.png`: Euler determinista versus dos trayectorias de Euler–Maruyama.
- `datos/`: CSV para todas las trayectorias y trenes de spikes, más `resultados.json` con indicadores reproducibles.
- `informe/informe.pdf`: informe breve de **máximo cuatro páginas**, con `informe.tex` editable.

## Fidelidad a las fuentes

- **Parte I:** parámetros, amplitudes y tiempos de corriente tomados literalmente del enunciado. Las corrientes discontinuas permanecen constantes durante cada paso RK4, para no mezclar dos tramos en la última etapa de Runge–Kutta. El valor `+30 mV` sólo registra un pico, **no** es un umbral fisiológico fijo.
- **Parte II:** red de `800` excitatorias y `200` inhibitorias, excitación/inhibición heterogéneas, matriz aleatoria densa `S`, ruido gaussiano nuevo **por milisegundo** y pulsos por eventos `v >= 30`. La ejecución con y sin diagonal comparte aleatoriedad (semilla `2026`). No se usa RK4 en esta reproducción porque el artículo exige otro esquema de actualización.
- **Extensión:** `dV=f dt + σ dW`, `dU=g dt`, con `ΔW = sqrt(h) ξ`; `σ=5` y `h=0.01 ms`, manteniendo la corriente escalón de la parte I para comparar. **No es** el ruido gaussiano renovado cada milisegundo del algoritmo Matlab, que representa otra discretización.
- La modificación opcional de la ley continua `z(v)` del final del enunciado **no** está incluida: es un modelo alternativo, no parte de la Figura 3 original.

Los parámetros/condiciones de los protocolos `TC1`, `TC2` y `RZ` provienen del práctico y no deben atribuirse directamente como datos tabulados al artículo de 2003. Las simulaciones permiten comparaciones **cualitativas**, no ajustes cuantitativos a registros biológicos.
