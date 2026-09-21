# rk4lab --- Integrador Runge--Kutta de cuarto orden en Python

`rk4lab` es un proyecto didáctico y reutilizable para resolver
**problemas de valores iniciales de ecuaciones diferenciales ordinarias
(EDO)** mediante el método clásico de **Runge--Kutta de cuarto orden
(RK4)**.

El proyecto fue pensado como base para los trabajos prácticos de
**Fundamentos de Redes Neuronales --- FAMAF 2026**. La idea central es
separar claramente:

``` text
modelo matemático  ──►  integrador RK4  ──►  solución
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         ▼                     ▼                     ▼
                      análisis              gráficos             comparación
                   (errores, fase,        temporales,            entre casos
                     Poincaré)          espacio de fase
```

El integrador **no sabe qué representa físicamente el problema**. Sólo
recibe una función de la forma

$$
\boxed{\frac{d\mathbf Y}{dt}=\mathbf F(t,\mathbf Y,\mathbf p)}
$$

donde $\mathbf Y$ es el vector de estado y $\mathbf p$ representa los
parámetros del modelo. Por eso el mismo núcleo sirve para una EDO
escalar, sistemas acoplados, EDO de orden superior transformadas a
primer orden, ecuaciones no lineales y modelos con una o varias entradas
externas.

------------------------------------------------------------------------

## Índice

-   [1. Runge--Kutta: idea general](#1-rungekutta-idea-general)
-   [2. RK4 clásico](#2-rk4-clásico)
-   [3. RK4 es un método explícito](#3-rk4-es-un-método-explícito)
-   [4. Sistemas de EDO y formulación
    vectorial](#4-sistemas-de-edo-y-formulación-vectorial)
-   [5. EDO de orden superior](#5-edo-de-orden-superior)
-   [6. Error local, error global y orden del
    método](#6-error-local-error-global-y-orden-del-método)
-   [7. Estructura del proyecto](#7-estructura-del-proyecto)
-   [8. Instalación](#8-instalación)
-   [9. Uso básico](#9-uso-básico)
-   [10. Entradas externas](#10-entradas-externas)
-   [11. Problemas de prueba
    incluidos](#11-problemas-de-prueba-incluidos)
-   [12. Análisis y gráficos](#12-análisis-y-gráficos)
-   [13. Tests automáticos](#13-tests-automáticos)
-   [14. Alcance y limitaciones](#14-alcance-y-limitaciones)
-   [15. Preparación para el TP
    neuronal](#15-preparación-para-el-tp-neuronal)

------------------------------------------------------------------------

## 1. Runge--Kutta: idea general

Los métodos de **Runge--Kutta (RK)** forman una familia de métodos
numéricos para aproximar la solución de un problema de valores iniciales

$$
\frac{dy}{dt}=f(t,y),
\qquad
y(t_0)=y_0.
$$

Se construye una sucesión de tiempos

$$
t_n=t_0+nh,
$$

donde $h$ es el **paso de integración**, y se busca aproximar

$$
y_n\simeq y(t_n).
$$

Si conocemos $(t_n,y_n)$, queremos construir una aproximación de
$y(t_n+h)$. La idea de los métodos RK es **evaluar la pendiente de la
solución en uno o varios puntos del intervalo de integración y combinar
esas evaluaciones para avanzar al siguiente tiempo**.

### RK1: método de Euler explícito

El método más sencillo utiliza solamente la pendiente al comienzo del
intervalo:

$$
k_1=f(t_n,y_n),
$$

$$
\boxed{y_{n+1}=y_n+h\,k_1.}
$$

Euler explícito puede interpretarse como un **Runge--Kutta de primer
orden (RK1)**.

### RK2: una evaluación intermedia

Una forma habitual de RK2 calcula

$$
k_1=f(t_n,y_n),
$$

$$
k_2=f\left(t_n+\frac h2,\;y_n+\frac h2k_1\right),
$$

y avanza mediante

$$
y_{n+1}=y_n+h\,k_2.
$$

La idea se generaliza: aumentar el orden requiere elegir adecuadamente
las evaluaciones intermedias y sus pesos.

> **Runge--Kutta no designa un único algoritmo.** RK1, RK2, RK3, RK4,
> etc. son miembros diferentes de la misma familia.

------------------------------------------------------------------------

## 2. RK4 clásico

Para

$$
\dot{\mathbf Y}=\mathbf F(t,\mathbf Y),
\qquad
\mathbf Y(t_0)=\mathbf Y_0,
$$

el RK4 clásico avanza desde $(t_n,\mathbf Y_n)$ hasta $t_{n+1}=t_n+h$
calculando cuatro pendientes:

$$
\mathbf k_1=\mathbf F(t_n,\mathbf Y_n),
$$

$$
\mathbf k_2=
\mathbf F\left(
t_n+\frac h2,\;
\mathbf Y_n+\frac h2\mathbf k_1
\right),
$$

$$
\mathbf k_3=
\mathbf F\left(
t_n+\frac h2,\;
\mathbf Y_n+\frac h2\mathbf k_2
\right),
$$

$$
\mathbf k_4=
\mathbf F\left(
t_n+h,\;
\mathbf Y_n+h\mathbf k_3
\right).
$$

Finalmente,

$$
\boxed{
\mathbf Y_{n+1}
=
\mathbf Y_n+
\frac h6
\left(
\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4
\right)
}
$$

Las cuatro evaluaciones corresponden, aproximadamente, al inicio, dos
estimaciones en el centro y el final del intervalo. La combinación final
es un **promedio ponderado** de esas pendientes: las dos evaluaciones
centrales tienen peso doble.

La implementación está en
[`src/rk4lab/integradores.py`](src/rk4lab/integradores.py),
principalmente en las funciones
[`paso_rk4`](src/rk4lab/integradores.py#L23-L40) y
[`resolver_rk4`](src/rk4lab/integradores.py#L43-L75).

------------------------------------------------------------------------

## 3. RK4 es un método explícito

Que $\mathbf Y$ sea un vector **no significa que haya que resolver un
sistema algebraico en cada paso**.

En RK4 clásico las etapas se evalúan secuencialmente:

``` text
Y_n ──► k1 ──► k2 ──► k3 ──► k4 ──► Y_(n+1)
```

Cuando se calcula $\mathbf k_1$, todas las cantidades son conocidas. Con
$\mathbf k_1$ se construye el estado intermedio necesario para
$\mathbf k_2$; luego se hace lo mismo con $\mathbf k_3$ y $\mathbf k_4$.

Por ejemplo, para

$$
\mathbf Y=
\begin{pmatrix}
x\\v
\end{pmatrix},
\qquad
\mathbf F(t,\mathbf Y)=
\begin{pmatrix}
v\\-\omega^2x
\end{pmatrix},
$$

la primera etapa es simplemente

$$
\mathbf k_1=
\begin{pmatrix}
v_n\\-\omega^2x_n
\end{pmatrix}.
$$

No hay incógnitas nuevas que despejar. Por eso **no se necesita
Gauss--Seidel, inversión de matrices ni Newton--Raphson** para ejecutar
el RK4 implementado aquí.

Esto contrasta con un método implícito. Por ejemplo, Euler implícito
escribe

$$
\mathbf Y_{n+1}
=
\mathbf Y_n+
h\mathbf F(t_{n+1},\mathbf Y_{n+1}),
$$

donde $\mathbf Y_{n+1}$ aparece dentro de $\mathbf F$. Allí sí puede ser
necesario resolver un sistema algebraico lineal o no lineal.

Si, por ejemplo,

$$
\dot{\mathbf Y}=A\mathbf Y,
$$

Euler implícito conduce a

$$
(I-hA)\mathbf Y_{n+1}=\mathbf Y_n,
$$

que sí es un sistema lineal. En un problema no lineal puede aparecer una
ecuación del tipo $\mathbf G(\mathbf Y_{n+1})=0$, para la cual podrían
utilizarse métodos como Newton--Raphson.

------------------------------------------------------------------------

## 4. Sistemas de EDO y formulación vectorial

Un sistema como

$$
\begin{cases}
\dot x=f_1(t,x,y,z),\\
\dot y=f_2(t,x,y,z),\\
\dot z=f_3(t,x,y,z)
\end{cases}
$$

se escribe de forma compacta definiendo

$$
\mathbf Y=
\begin{pmatrix}
x\\y\\z
\end{pmatrix},
\qquad
\mathbf F(t,\mathbf Y)=
\begin{pmatrix}
f_1(t,\mathbf Y)\\
f_2(t,\mathbf Y)\\
f_3(t,\mathbf Y)
\end{pmatrix}.
$$

Así,

$$
\boxed{\dot{\mathbf Y}=\mathbf F(t,\mathbf Y).}
$$

Cada $\mathbf k_i$ de RK4 es entonces un vector de la misma dimensión
que $\mathbf Y$.

En NumPy, la traducción es casi literal:

``` python
k1 = F(t, Y)
k2 = F(t + h/2, Y + h*k1/2)
k3 = F(t + h/2, Y + h*k2/2)
k4 = F(t + h,   Y + h*k3)

Y_nuevo = Y + h*(k1 + 2*k2 + 2*k3 + k4)/6
```

### Caso lineal matricial

Si el sistema tiene la forma

$$
\dot{\mathbf Y}=A\mathbf Y+B\mathbf u(t),
$$

puede escribirse en Python como

``` python
def modelo(t, Y, parametros):
    A = parametros["A"]
    B = parametros["B"]
    u = parametros["entrada"](t)
    return A @ Y + B @ u
```

El operador `@` realiza el producto matricial. **El integrador RK4 no
cambia**: la estructura matricial pertenece al modelo $\mathbf F$, no al
algoritmo de integración.

### Caso no lineal

Para

$$
\begin{cases}
\dot x=y-x^3,\\
\dot y=\sin x-xy,
\end{cases}
$$

simplemente se define

``` python
def modelo(t, Y, parametros):
    x, y = Y
    return np.array([
        y - x**3,
        np.sin(x) - x*y,
    ])
```

La misma función `resolver_rk4` integra ambos casos.

------------------------------------------------------------------------

## 5. EDO de orden superior

RK4 se aplica a sistemas de primer orden. Una EDO de orden superior se
transforma introduciendo variables auxiliares.

### Ejemplo: oscilador armónico

Partimos de

$$
\ddot x+\omega^2x=0.
$$

Definimos

$$
y_1=x,
\qquad
y_2=\dot x=v.
$$

Entonces

$$
\dot y_1=y_2,
\qquad
\dot y_2=-\omega^2y_1,
$$

y por tanto

$$
\boxed{
\dot{\mathbf Y}
=
\begin{pmatrix}
y_2\\
-\omega^2y_1
\end{pmatrix},
\qquad
\mathbf Y=
\begin{pmatrix}
y_1\\y_2
\end{pmatrix}.
}
$$

La implementación utilizada en los ejemplos puede verse en
[`oscilador_armonico`](src/rk4lab/modelos.py#L16-L21).

### Ejemplo: oscilador amortiguado y forzado

Para

$$
\ddot x+\beta\dot x+\omega^2x=F(t),
$$

se obtiene

$$
\begin{cases}
\dot y_1=y_2,\\
\dot y_2=F(t)-\beta y_2-\omega^2y_1.
\end{cases}
$$

### Caso general de orden (m)

Si

$$
y^{(m)}
=
f\left(t,y,y',y'',\ldots,y^{(m-1)}\right),
$$

se define

$$
y_1=y,\qquad
y_2=y',\qquad
y_3=y'',\quad\ldots,\quad
y_m=y^{(m-1)}.
$$

Entonces

$$
\begin{aligned}
\dot y_1 &= y_2,\\
\dot y_2 &= y_3,\\
&\vdots\\
\dot y_{m-1} &= y_m,\\
\dot y_m &= f(t,y_1,y_2,\ldots,y_m).
\end{aligned}
$$

Para el integrador, **una EDO de cuarto orden convertida en cuatro
ecuaciones y cuatro EDO originalmente acopladas son el mismo tipo de
problema numérico**: un sistema vectorial de primer orden.

------------------------------------------------------------------------

## 6. Error local, error global y orden del método

El nombre **"cuarto orden"** no significa "cuatro decimales correctos"
ni se debe simplemente a que RK4 evalúe cuatro pendientes.

La solución exacta admite un desarrollo de Taylor:

$$
y(t+h)
=
y(t)+hy'(t)
+\frac{h^2}{2!}y''(t)
+\frac{h^3}{3!}y'''(t)
+\frac{h^4}{4!}y^{(4)}(t)
+O(h^5).
$$

Al desarrollar las etapas de RK4 y sustituirlas en la combinación final,
el método reproduce los términos hasta orden $h^4$. La primera
discrepancia aparece en orden $h^5$.

Por eso:

$$
\boxed{E_{\mathrm{local}}=O(h^5)}
$$

y, para un tiempo final fijo,

$$
\boxed{E_{\mathrm{global}}=O(h^4).}
$$

### Error local

El **error local** responde a la pregunta:

> Si comienzo un único paso desde el valor exacto, ¿cuánto error
> introduce ese paso de RK4?

Esquemáticamente,

``` text
valor exacto en t_n ──[un paso RK4]──► aproximación en t_(n+1)
```

Para RK4, ese error es de orden $h^5$.

### Error global

En una integración real, salvo el dato inicial, los pasos posteriores
comienzan desde valores que ya son aproximados:

``` text
Y_0 exacto ─► Y_1 RK4 ─► Y_2 RK4 ─► ... ─► Y_N RK4
```

El **error global** compara el valor numérico final con la solución
exacta en ese mismo tiempo.

Para un intervalo de longitud $T$,

$$
N\approx\frac{T}{h}.
$$

Como intuición,

$$
E_{\mathrm{global}}
\sim
N\,O(h^5)
\sim
\frac{T}{h}O(h^5)
=
O(Th^4).
$$

Para $T$ fijo:

$$
\boxed{E_{\mathrm{global}}=O(h^4).}
$$

Esta cuenta explica intuitivamente la pérdida de una potencia. La
demostración rigurosa requiere además hipótesis de regularidad y
estabilidad.

### Verificación numérica del orden

El proyecto verifica estos órdenes con

$$
\dot y=y,\qquad y(0)=1,
$$

cuya solución exacta es

$$
y(t)=e^t.
$$

Para el error local se realiza **un solo paso** de tamaño $h$ desde el
dato exacto. Para el error global se integra siempre hasta el mismo
tiempo final $T=3$.

Al reducir sucesivamente

$$
h,\quad \frac h2,\quad \frac h4,\quad \frac h8,\ldots,
$$

se calcula el orden observado

$$
\boxed{
p=
\log_2\left(\frac{E(h)}{E(h/2)}\right).
}
$$

Si $E(h)\propto h^q$, entonces $p\to q$. Por lo tanto esperamos

$$
p_{\mathrm{local}}\to5,
\qquad
p_{\mathrm{global}}\to4.
$$

El código del experimento está en el [ejemplo de
convergencia](examples/ejecutar_ejemplos.py#L71-L88), mientras que el
cálculo de errores se implementa en
[`estudio_orden_rk4`](src/rk4lab/analisis.py#L18-L40).

```{=html}
<p align="center">
```
`<img src="figuras/06_convergencia_rk4.png" alt="Convergencia de RK4" width="720">`{=html}
`<br>`{=html} `<em>`{=html}Figura 1. Verificación numérica del orden de
RK4. En escala log-log, el error local sigue una referencia proporcional
a h⁵ y el error global una referencia proporcional a h⁴.`</em>`{=html}
```{=html}
</p>
```
Una consecuencia práctica es que, en el régimen asintótico, al
reemplazar $h$ por $h/2$:

$$
E_{\mathrm{local}}(h/2)\approx\frac{E_{\mathrm{local}}(h)}{2^5}
=\frac{E_{\mathrm{local}}(h)}{32},
$$

mientras que

$$
E_{\mathrm{global}}(h/2)\approx\frac{E_{\mathrm{global}}(h)}{2^4}
=\frac{E_{\mathrm{global}}(h)}{16}.
$$

------------------------------------------------------------------------

## 7. Estructura del proyecto

``` text
rk4/
├── README.md
├── pyproject.toml
├── examples/
│   └── ejecutar_ejemplos.py
├── figuras/
│   ├── 01_decaimiento_exponencial.png
│   ├── 02_oscilador_fase.png
│   ├── 03_dos_entradas.png
│   ├── 04_duffing_poincare.png
│   ├── 05_lorenz.png
│   └── 06_convergencia_rk4.png
├── src/
│   └── rk4lab/
│       ├── __init__.py
│       ├── integradores.py
│       ├── modelos.py
│       ├── senales.py
│       ├── analisis.py
│       └── graficos.py
└── tests/
    └── test_rk4.py
```

La separación es intencional:

-   [`integradores.py`](src/rk4lab/integradores.py): núcleo numérico,
    `paso_rk4`, `resolver_rk4` y `Solucion`.
-   [`modelos.py`](src/rk4lab/modelos.py): ecuaciones diferenciales
    utilizadas como modelos.
-   [`senales.py`](src/rk4lab/senales.py): entradas externas
    reutilizables.
-   [`analisis.py`](src/rk4lab/analisis.py): sección de Poincaré,
    errores y orden observado.
-   [`graficos.py`](src/rk4lab/graficos.py): gráficos temporales,
    espacio de fase y convergencia.
-   [`ejecutar_ejemplos.py`](examples/ejecutar_ejemplos.py): problemas
    de prueba y generación de figuras.
-   [`test_rk4.py`](tests/test_rk4.py): pruebas automáticas.

La filosofía es mantener separadas las responsabilidades de **modelar,
integrar, analizar y visualizar**.

------------------------------------------------------------------------

## 8. Instalación

Desde la carpeta `practicos/integradores/rk4/`:

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

En Windows, la activación del entorno virtual puede hacerse desde
PowerShell con:

``` powershell
.venv\Scripts\Activate.ps1
```

La instalación editable permite modificar el código de `src/rk4lab/` sin
reinstalar el paquete después de cada cambio.

### Ejecutar los tests

``` bash
pytest -q
```

### Ejecutar todos los ejemplos y regenerar las figuras

``` bash
python examples/ejecutar_ejemplos.py
```

------------------------------------------------------------------------

## 9. Uso básico

Todo modelo debe respetar la interfaz

``` python
def modelo(t, y, parametros):
    return derivadas
```

donde:

-   `t` es el tiempo actual;
-   `y` es un `numpy.ndarray` con el estado actual;
-   `parametros` contiene constantes, entradas u otros datos del modelo;
-   el valor devuelto debe ser un vector con la **misma dimensión que
    `y`**.

### Ejemplo mínimo

Para

$$
\dot y=-2y,
\qquad
y(0)=1,
$$

podemos escribir:

``` python
import numpy as np
from rk4lab.integradores import resolver_rk4


def modelo(t, y, parametros):
    return np.array([-2.0 * y[0]])


solucion = resolver_rk4(
    funcion=modelo,
    intervalo_t=(0, 5),
    y0=[1.0],
    dt=0.01,
    parametros={},
    nombres=["y"],
)

print(solucion.t)
print(solucion.y[:, 0])
```

`solucion.t` contiene los tiempos de integración y `solucion.y` tiene
forma

``` text
(numero_de_tiempos, numero_de_variables)
```

Por ejemplo,

``` python
solucion.y[:, 0]
```

selecciona la evolución temporal de la primera variable.

También puede utilizarse

``` python
solucion.estado(0)
```

para obtener esa misma componente.

------------------------------------------------------------------------

## 10. Entradas externas

Una entrada externa se representa como una función del tiempo. Las
señales disponibles se encuentran en
[`senales.py`](src/rk4lab/senales.py):

-   [`constante`](src/rk4lab/senales.py#L5-L6)
-   [`sinusoidal`](src/rk4lab/senales.py#L9-L11)
-   [`pulso`](src/rk4lab/senales.py#L14-L15)
-   [`sumar_senales`](src/rk4lab/senales.py#L18-L19)

Por ejemplo:

``` python
from rk4lab.senales import sinusoidal, pulso

entrada_1 = sinusoidal(amplitud=1.0, frecuencia=0.5)
entrada_2 = pulso(amplitud=2.0, t_inicio=3.0, t_fin=6.0)
```

Un modelo con dos entradas simultáneas puede ser

$$
\dot y=-ay+b_1u_1(t)+b_2u_2(t),
$$

y escribirse como

``` python
def modelo(t, y, p):
    u1 = p["entrada1"](t)
    u2 = p["entrada2"](t)

    return np.array([
        -p["a"]*y[0]
        + p["b1"]*u1
        + p["b2"]*u2
    ])
```

La cantidad de entradas **no modifica RK4**. Si el futuro TP requiere 1,
2 u 8 entradas, esa estructura pertenece al modelo y a sus parámetros.

```{=html}
<p align="center">
```
`<img src="figuras/03_dos_entradas.png" alt="Dos entradas simultáneas y una salida" width="720">`{=html}
`<br>`{=html} `<em>`{=html}Figura 2. Ejemplo de una variable de estado
excitada simultáneamente por una entrada sinusoidal y un pulso. El
integrador sólo recibe la derivada resultante; no necesita conocer
cuántas entradas posee el modelo.`</em>`{=html}
```{=html}
</p>
```
Código: [ejemplo 3 --- dos entradas
simultáneas](examples/ejecutar_ejemplos.py#L39-L52).

------------------------------------------------------------------------

## 11. Problemas de prueba incluidos

Los ejemplos se encuentran en
[`examples/ejecutar_ejemplos.py`](examples/ejecutar_ejemplos.py). Cada
uno verifica una característica diferente del integrador.

### 11.1 Decaimiento exponencial --- EDO escalar

Se resuelve

$$
\dot y=-\lambda y,
\qquad
y(0)=1,
$$

cuya solución exacta para $\lambda=1$ es

$$
y(t)=e^{-t}.
$$

Este caso permite comparar directamente la solución numérica contra una
solución analítica conocida.

**Código:** [ejemplo 1](examples/ejecutar_ejemplos.py#L24-L30) ·
**Modelo:** [`decaimiento_exponencial`](src/rk4lab/modelos.py#L10-L13)

```{=html}
<p align="center">
```
`<img src="figuras/01_decaimiento_exponencial.png" alt="Decaimiento exponencial" width="720">`{=html}
`<br>`{=html} `<em>`{=html}Figura 3. Comparación entre la solución
obtenida con RK4 y la solución exacta del decaimiento
exponencial.`</em>`{=html}
```{=html}
</p>
```
### 11.2 Oscilador armónico --- EDO de segundo orden

Se resuelve

$$
\ddot x+\omega^2x=0
$$

transformándola en el sistema

$$
\dot x=v,
\qquad
\dot v=-\omega^2x.
$$

Este ejemplo verifica la formulación vectorial y la transformación de
una EDO de orden superior.

**Código:** [ejemplo 2](examples/ejecutar_ejemplos.py#L32-L37) ·
**Modelo:** [`oscilador_armonico`](src/rk4lab/modelos.py#L16-L21)

```{=html}
<p align="center">
```
`<img src="figuras/02_oscilador_fase.png" alt="Espacio de fase del oscilador armónico" width="720">`{=html}
`<br>`{=html} `<em>`{=html}Figura 4. Trayectoria del oscilador armónico
en el espacio de fase (x,v).`</em>`{=html}
```{=html}
</p>
```
### 11.3 Dos entradas simultáneas

Se estudia

$$
\dot y=-ay+b_1u_1(t)+b_2u_2(t),
$$

con una señal sinusoidal y un pulso aplicados al mismo tiempo.

**Código:** [ejemplo 3](examples/ejecutar_ejemplos.py#L39-L52) ·
**Modelo:**
[`sistema_lineal_dos_entradas`](src/rk4lab/modelos.py#L24-L33)

La figura correspondiente se mostró en la [sección de entradas
externas](#10-entradas-externas).

### 11.4 Oscilador de Duffing --- no linealidad y Poincaré

El oscilador de Duffing utilizado tiene la forma

$$
\ddot x+\delta\dot x+\alpha x+\beta x^3=F(t).
$$

Se transforma en un sistema de dos EDO de primer orden y se integra con
RK4. Como el forzado es periódico, se construye además una **sección de
Poincaré** muestreando el estado una vez por período después de
descartar un transitorio.

**Código:** [ejemplo 4](examples/ejecutar_ejemplos.py#L54-L63) ·
**Modelo:** [`duffing`](src/rk4lab/modelos.py#L36-L43) · **Análisis:**
[`seccion_poincare`](src/rk4lab/analisis.py#L6-L15)

```{=html}
<p align="center">
```
`<img src="figuras/04_duffing_poincare.png" alt="Sección de Poincaré del oscilador de Duffing" width="720">`{=html}
`<br>`{=html} `<em>`{=html}Figura 5. Sección de Poincaré del oscilador
de Duffing forzado, obtenida muestreando la trayectoria una vez por
período del forzado.`</em>`{=html}
```{=html}
</p>
```
### 11.5 Sistema de Lorenz --- tres EDO no lineales acopladas

Se integra el sistema

$$
\begin{aligned}
\dot x &= \sigma(y-x),\\
\dot y &= x(\rho-z)-y,\\
\dot z &= xy-\beta z.
\end{aligned}
$$

Este ejemplo verifica que el mismo integrador funciona con un estado
vectorial tridimensional y ecuaciones no lineales acopladas.

**Código:** [ejemplo 5](examples/ejecutar_ejemplos.py#L65-L69) ·
**Modelo:** [`lorenz`](src/rk4lab/modelos.py#L46-L57)

```{=html}
<p align="center">
```
`<img src="figuras/05_lorenz.png" alt="Proyección x-z del sistema de Lorenz" width="720">`{=html}
`<br>`{=html} `<em>`{=html}Figura 6. Proyección x-z de la trayectoria
obtenida para el sistema de Lorenz.`</em>`{=html}
```{=html}
</p>
```
### 11.6 Convergencia --- orden local 5 y global 4

El problema

$$
\dot y=y,\qquad y(0)=1,\qquad y(t)=e^t
$$

se utiliza para medir experimentalmente los errores local y global al
reducir $h$ por factores de dos.

**Código:** [ejemplo 6](examples/ejecutar_ejemplos.py#L71-L88) ·
**Análisis:** [`estudio_orden_rk4`](src/rk4lab/analisis.py#L18-L40) y
[`orden_observado`](src/rk4lab/analisis.py#L43-L46)

La figura correspondiente se mostró en la [sección de
errores](#6-error-local-error-global-y-orden-del-método).

### 11.7 Problema sin unicidad --- un caso patológico útil

Consideremos

$$
\dot y=2\sqrt y,
\qquad
y(0)=0.
$$

Para $t\ge0$, tanto

$$
y(t)=0
$$

como

$$
y(t)=t^2
$$

satisfacen el problema, y existen además soluciones que permanecen un
tiempo en cero antes de comenzar a crecer.

Si RK4 comienza exactamente en $y_0=0$,

$$
k_1=k_2=k_3=k_4=0,
$$

por lo que permanece en la solución $y(t)=0$ aunque se reduzca el paso.

Esto **no es un fallo del integrador**: el problema de valores iniciales
no determina una solución única.

**Código:** [ejemplo 7](examples/ejecutar_ejemplos.py#L90-L93) ·
**Modelo:** [`raiz_no_unica`](src/rk4lab/modelos.py#L67-L69)

------------------------------------------------------------------------

## 12. Análisis y gráficos

El integrador se mantiene separado de las herramientas de análisis y
visualización.

### Evolución temporal

``` python
from rk4lab.graficos import graficar_temporal

ax = graficar_temporal(solucion)
```

Código: [`graficar_temporal`](src/rk4lab/graficos.py#L5-L13).

### Espacio de fase

Para dos componentes $Y_i$ y $Y_j$:

``` python
from rk4lab.graficos import graficar_fase

ax = graficar_fase(solucion, i=0, j=1)
```

Código: [`graficar_fase`](src/rk4lab/graficos.py#L16-L23).

### Sección de Poincaré

Para una entrada periódica de período $T$:

``` python
from rk4lab.analisis import seccion_poincare

tiempos, puntos = seccion_poincare(
    solucion,
    periodo=T,
    t_inicio=100.0,
)
```

Código: [`seccion_poincare`](src/rk4lab/analisis.py#L6-L15).

### Estudio de convergencia

``` python
from rk4lab.analisis import estudio_orden_rk4, orden_observado
```

Estas herramientas permiten comprobar numéricamente que la
implementación conserva el orden esperado del RK4 clásico.

------------------------------------------------------------------------

## 13. Tests automáticos

Las pruebas se encuentran en [`tests/test_rk4.py`](tests/test_rk4.py) y
verifican:

1.  el decaimiento exponencial frente a su solución exacta;
2.  la integración vectorial del oscilador armónico;
3.  un paso conocido de RK4 para $\dot y=y$;
4.  los órdenes observados $p_{\rm local}\approx5$ y
    $p_{\rm global}\approx4$;
5.  el comportamiento del problema no único $\dot y=2\sqrt y$.

Se ejecutan con

``` bash
pytest -q
```

La prueba del orden es especialmente importante: no sólo comprueba que
el resultado "parece correcto", sino que verifica una propiedad
matemática característica del algoritmo implementado.

------------------------------------------------------------------------

## 14. Alcance y limitaciones

### EDO

El integrador resuelve directamente problemas de la forma

$$
\dot{\mathbf Y}=\mathbf F(t,\mathbf Y,\mathbf p).
$$

### EDP

Una ecuación diferencial parcial no se integra directamente con este
RK4. En ciertos problemas puede discretizarse primero el espacio para
obtener un sistema de EDO:

``` text
EDP ──► discretización espacial ──► sistema de EDO ──► integración temporal
```

RK4 puede entonces utilizarse para la integración temporal si el
problema resultante lo permite.

### Ecuaciones con retardo

Una ecuación del tipo

$$
\dot y(t)=f\bigl(t,y(t),y(t-\tau)\bigr)
$$

requiere almacenar e interpolar la historia de la solución. El RK4
implementado aquí no incorpora esa funcionalidad.

### Ecuaciones diferenciales estocásticas

Problemas que incluyen términos aleatorios, por ejemplo

$$
dY=f(Y,t)\,dt+\sigma\,dW_t,
$$

requieren métodos diseñados para ecuaciones diferenciales estocásticas.

### Sistemas rígidos

RK4 es explícito. En problemas **rígidos (stiff)**, la estabilidad puede
obligar a utilizar pasos extremadamente pequeños. En esos casos pueden
ser más adecuados métodos implícitos o especializados, como BDF o Radau.

------------------------------------------------------------------------

## 15. Preparación para el TP neuronal

El proyecto deja preparado el núcleo para que el futuro modelo neuronal
pueda incorporarse sin modificar el integrador.

En [`modelos.py`](src/rk4lab/modelos.py) ya existe una dinámica
subumbral **LIF (Leaky Integrate-and-Fire)**:

``` python
def neurona_lif_subumbral(t, y, p):
    V = y[0]
    corriente = p.get("entrada", lambda _t: 0.0)(t)

    return np.array([
        (-(V - p["E_L"]) + p["R_m"] * corriente) / p["tau_m"]
    ])
```

El disparo y el reset son eventos discretos y se mantienen fuera del
núcleo RK4 hasta conocer el modelo exacto requerido por el trabajo
práctico.

La arquitectura permite agregar posteriormente:

-   una o varias señales de entrada;
-   uno o varios potenciales o estados de salida;
-   barridos de parámetros o estímulos;
-   comparación entre múltiples experimentos;
-   gráficos de entrada y salida;
-   espacio de fase;
-   secciones de Poincaré;
-   medidas de error y convergencia.

La interfaz que debe permanecer estable es

$$
\boxed{
\dot{\mathbf Y}
=
\mathbf F(t,\mathbf Y,\mathbf p).
}
$$

De esta forma, **el modelo físico puede cambiar sin reescribir el
integrador numérico**.

------------------------------------------------------------------------

## Referencia rápida

``` python
from rk4lab.integradores import resolver_rk4

solucion = resolver_rk4(
    funcion=modelo,
    intervalo_t=(t0, tf),
    y0=estado_inicial,
    dt=paso,
    parametros=parametros,
    nombres=nombres,
)
```

Para ejecutar el proyecto completo:

``` bash
pytest -q
python examples/ejecutar_ejemplos.py
```

El núcleo RK4 permanece deliberadamente pequeño; la versatilidad
proviene de representar cada problema mediante una función
$\mathbf F(t,\mathbf Y,\mathbf p)$ y mantener separados el **modelo**,
la **integración**, el **análisis** y la **visualización**.
