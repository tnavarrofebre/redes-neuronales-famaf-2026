# rk4lab — Integrador Runge–Kutta 4 en Python

Proyecto didáctico y reutilizable para integrar **ecuaciones diferenciales ordinarias (EDO)** mediante el método clásico de **Runge–Kutta de cuarto orden (RK4)**. Está pensado como base para un trabajo práctico de sistemas dinámicos y modelos neuronales, pero el núcleo numérico no sabe qué representa físicamente el problema: recibe simplemente

\[
\frac{d\mathbf Y}{dt}=\mathbf F(t,\mathbf Y,\mathbf p).
\]

Por eso el mismo integrador sirve para una EDO escalar, un sistema acoplado, una EDO de orden superior transformada a primer orden, ecuaciones no lineales y modelos con una o varias entradas externas.

## 1. Idea matemática

Para el problema de valores iniciales

\[
\dot{\mathbf Y}=\mathbf F(t,\mathbf Y),\qquad \mathbf Y(t_0)=\mathbf Y_0,
\]

RK4 avanza desde \((t_n,\mathbf Y_n)\) hasta \(t_{n+1}=t_n+h\) evaluando cuatro pendientes:

\[
\mathbf k_1=\mathbf F(t_n,\mathbf Y_n),
\]

\[
\mathbf k_2=\mathbf F\!\left(t_n+\frac h2,\mathbf Y_n+\frac h2\mathbf k_1\right),
\]

\[
\mathbf k_3=\mathbf F\!\left(t_n+\frac h2,\mathbf Y_n+\frac h2\mathbf k_2\right),
\]

\[
\mathbf k_4=\mathbf F(t_n+h,\mathbf Y_n+h\mathbf k_3),
\]

y combina esas pendientes mediante

\[
\boxed{\mathbf Y_{n+1}=\mathbf Y_n+\frac h6(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4)}.
\]

### RK4 es explícito

Aunque \(\mathbf Y\) sea un vector, **no hay que resolver un sistema algebraico en cada paso**. Cada etapa se calcula con cantidades ya conocidas:

```text
Y_n -> k1 -> k2 -> k3 -> k4 -> Y_(n+1)
```

Por ejemplo, si

\[
\mathbf Y=(x,v)^T,\qquad
\mathbf F(t,\mathbf Y)=(v,-\omega^2x)^T,
\]

entonces \(\mathbf k_1\) se obtiene evaluando directamente esas dos expresiones en \((t_n,x_n,v_n)\); luego \(\mathbf k_2\) se evalúa en el estado intermedio construido con \(\mathbf k_1\), etc. No se necesita Gauss–Seidel, inversión de matrices ni Newton–Raphson.

Esto contrasta con un método implícito. Euler implícito, por ejemplo,

\[
\mathbf Y_{n+1}=\mathbf Y_n+h\mathbf F(t_{n+1},\mathbf Y_{n+1}),
\]

contiene la incógnita \(\mathbf Y_{n+1}\) dentro de \(\mathbf F\), por lo que sí puede exigir resolver un sistema algebraico lineal o no lineal.

## 2. EDO de orden superior

RK4 se formula para sistemas de primer orden, pero una EDO de orden superior se transforma introduciendo variables auxiliares. Por ejemplo,

\[
\ddot x+\omega^2x=0
\]

se escribe definiendo \(v=\dot x\):

\[
\mathbf Y=\begin{pmatrix}x\\v\end{pmatrix},\qquad
\dot{\mathbf Y}=
\begin{pmatrix}
v\\-\omega^2x
\end{pmatrix}.
\]

En general, si

\[
y^{(m)}=f(t,y,y',\ldots,y^{(m-1)}),
\]

se define

\[
y_1=y,\ y_2=y',\ldots,\ y_m=y^{(m-1)},
\]

y se obtiene un sistema de \(m\) EDO de primer orden. Para el integrador, una EDO de cuarto orden transformada en cuatro variables y cuatro EDO originalmente acopladas son simplemente dos casos de la misma estructura vectorial.

## 3. Orden y errores

El RK4 clásico reproduce el desarrollo de Taylor de la solución hasta los términos de orden \(h^4\). Por ello:

\[
\boxed{E_{\mathrm{local}}=O(h^5)},\qquad
\boxed{E_{\mathrm{global}}=O(h^4)}.
\]

El **error local** mide el error producido por *un solo paso* suponiendo que ese paso comienza desde el valor exacto. El **error global** mide el error de \(\mathbf Y(t)\) después de avanzar desde el dato inicial mediante muchos pasos numéricos.

Para un intervalo temporal fijo \(T\), el número de pasos es aproximadamente \(N=T/h\). Intuitivamente, acumular \(N\sim 1/h\) errores locales de orden \(h^5\) explica la pérdida de una potencia y el comportamiento global \(h^4\). La demostración rigurosa requiere además hipótesis de regularidad y estabilidad.

### Verificación numérica incluida

El proyecto comprueba el orden usando

\[
y'=y,\qquad y(0)=1,\qquad y(t)=e^t.
\]

Para pasos \(h,h/2,h/4,\ldots\), se calcula

\[
p=\log_2\!\left(\frac{E(h)}{E(h/2)}\right).
\]

Se espera observar

\[
p_{\mathrm{local}}\to5,\qquad p_{\mathrm{global}}\to4.
\]

El ejemplo genera además `figuras/06_convergencia_rk4.png`, un gráfico log-log de ambos errores junto con referencias proporcionales a \(h^5\) y \(h^4\).

## 4. Estructura del proyecto

```text
rk4_neuron_project/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── rk4lab/
│       ├── __init__.py
│       ├── integradores.py   # paso_rk4 y resolver_rk4
│       ├── modelos.py        # ecuaciones diferenciales
│       ├── senales.py        # entradas externas
│       ├── analisis.py       # Poincare, errores y orden observado
│       └── graficos.py       # graficos temporales, fase y convergencia
├── examples/
│   └── ejecutar_ejemplos.py
├── tests/
│   └── test_rk4.py
└── figuras/
```

La separación es intencional: **integrar**, **definir el modelo**, **generar entradas**, **analizar** y **graficar** son responsabilidades distintas. Así puede cambiarse el modelo neuronal sin tocar RK4.

## 5. Instalación

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

En Windows, la activación del entorno virtual cambia por el comando correspondiente de PowerShell o `cmd`.

Para ejecutar las pruebas automáticas:

```bash
pytest -q
```

Para ejecutar todos los ejemplos y regenerar las figuras:

```bash
python examples/ejecutar_ejemplos.py
```

## 6. Uso mínimo

Un modelo debe ser una función con la interfaz

```python
def modelo(t, y, parametros):
    return derivadas
```

Por ejemplo, para

\[
y'=-2y,\qquad y(0)=1,
\]

puede escribirse:

```python
import numpy as np
from rk4lab.integradores import resolver_rk4


def modelo(t, y, parametros):
    return np.array([-2.0 * y[0]])


solucion = resolver_rk4(
    modelo,
    intervalo_t=(0, 5),
    y0=[1.0],
    dt=0.01,
    parametros={},
    nombres=["y"],
)

print(solucion.t)
print(solucion.y[:, 0])
```

`solucion.t` contiene los tiempos y `solucion.y` tiene forma

```text
(numero_de_tiempos, numero_de_variables)
```

por lo que `solucion.y[:, 0]` es la primera variable en todos los tiempos.

## 7. Ejemplo de segundo orden

Para

\[
\ddot x+\omega^2x=0,
\]

se usa \(\mathbf Y=(x,v)\):

```python
import numpy as np


def oscilador(t, y, parametros):
    x, v = y
    omega = parametros["omega"]
    return np.array([
        v,
        -omega**2 * x,
    ])
```

Luego se integra exactamente igual que una EDO escalar:

```python
solucion = resolver_rk4(
    oscilador,
    (0, 20),
    [1.0, 0.0],
    0.01,
    {"omega": 1.0},
    nombres=["x", "v"],
)
```

## 8. Una o varias entradas

Las entradas externas son funciones del tiempo independientes del integrador. Por ejemplo:

```python
from rk4lab.senales import sinusoidal, pulso

entrada_1 = sinusoidal(amplitud=1.0, frecuencia=0.5)
entrada_2 = pulso(amplitud=2.0, t_inicio=3.0, t_fin=6.0)
```

Un modelo puede usar ambas:

\[
y'=-ay+b_1u_1(t)+b_2u_2(t).
\]

```python
def modelo(t, y, p):
    u1 = p["entrada1"](t)
    u2 = p["entrada2"](t)
    return np.array([-p["a"]*y[0] + p["b1"]*u1 + p["b2"]*u2])
```

RK4 no cambia si posteriormente el TP requiere 1, 2 u 8 entradas: esa información pertenece al modelo y a sus parámetros, no al algoritmo de integración.

## 9. Problemas incluidos

`examples/ejecutar_ejemplos.py` contiene varios problemas con objetivos distintos:

1. **Decaimiento exponencial:** una EDO escalar y comparación con solución exacta.
2. **Oscilador armónico:** EDO de segundo orden convertida en un sistema de dos EDO; se grafica el espacio de fase.
3. **Dos entradas simultáneas:** una salida sometida a una entrada sinusoidal y a un pulso.
4. **Duffing forzado:** sistema no lineal, forzado periódico y sección de Poincaré.
5. **Lorenz:** tres variables acopladas y no lineales.
6. **Convergencia de RK4:** demuestra numéricamente los órdenes local 5 y global 4.
7. **Problema sin unicidad:** \(y'=2\sqrt y,\ y(0)=0\), usado para mostrar que la convergencia también depende de que el problema diferencial esté bien planteado.

En `modelos.py` queda además preparada la dinámica subumbral de una neurona **LIF (Leaky Integrate-and-Fire)**. El disparo y el reset son eventos discretos y se mantienen fuera del núcleo RK4 hasta conocer el modelo exacto pedido por el trabajo práctico.

## 10. El ejemplo no único: una advertencia útil

La ecuación

\[
y'=2\sqrt y,\qquad y(0)=0
\]

admite \(y(t)=0\) y también \(y(t)=t^2\) para \(t\ge0\), además de otras soluciones que permanecen un tiempo en cero antes de crecer. Al comenzar exactamente en \(y=0\), RK4 obtiene

\[
k_1=k_2=k_3=k_4=0
\]

y permanece en la solución \(y=0\), independientemente de cuánto se reduzca \(h\). Esto no es un fallo del integrador: el problema de valores iniciales no selecciona de manera única la rama \(t^2\).

## 11. Alcance

El RK4 clásico implementado aquí integra directamente **EDO**. Si aparece una EDP, una estrategia posible es discretizar primero las variables espaciales para obtener un sistema de EDO y luego integrar el tiempo; esa elección depende del problema. Las ecuaciones con retardo y las ecuaciones diferenciales estocásticas requieren herramientas adicionales.

También debe recordarse que RK4 explícito puede ser poco conveniente para problemas **rígidos (stiff)**, donde métodos implícitos o especializados pueden permitir pasos temporales mucho mayores sin perder estabilidad.

## 12. Filosofía para el futuro TP neuronal

La interfaz central seguirá siendo

\[
\boxed{\dot{\mathbf Y}=\mathbf F(t,\mathbf Y,\mathbf p)}.
\]

Cuando se conozca el enunciado del TP, la idea es agregar el modelo neuronal y sus entradas como nuevas funciones, conservando intacto el núcleo `paso_rk4` / `resolver_rk4`. Sobre las soluciones podrán añadirse gráficos de potencial de entrada y salida, múltiples estímulos, espacios de fase, secciones de Poincaré y comparaciones entre experimentos sin mezclar esas tareas con el método numérico.
