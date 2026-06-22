# Interpretación de Señales - TradingLatino HMM v4

## Formato de alerta Telegram

```
🔴 BTC-USD [1D] SHORT (inicial)
Precio: $64,239.68    Fuerza: 78%
Score LONG: 32 | SHORT: 78  (umbral: 60)
Regimen: Oso fuerte (conf 92%)
Previo: LONG (hace 3 velas)
TP -1.5%: $63,276.08 | Trail +1.0%: $64,882.08
Expira en 14 velas (~14 d)
─────
🤖 TradingLatino HMM v4
```

---

## 1. Dirección de la señal

### `🔴 BTC-USD [1D] SHORT (inicial)`

**Emoji `🔴` / `🟢`**
- 🔴 rojo = **SHORT** — el modelo detectó condiciones para vender (apuestas a que el precio baje).
- 🟢 verde = **LONG** — el modelo detectó condiciones para comprar (apuestas a que el precio suba).

**BTC-USD**
El activo analizado (Bitcoin vs Dólar estadounidense).

**[1D]** — Timeframe (tamaño de vela) en el que se generó la señal:
| Timeframe | Velas | Señal apta para... |
|-----------|-------|--------------------|
| `1H` | 1 hora | Scalping / corto plazo |
| `4H` | 4 horas | Intradía / medio plazo |
| `1D` | 1 día | Swing trading |
| `1W` | 1 semana | Largo plazo |

A mayor timeframe, más confiable pero más lenta en reaccionar.

**SHORT**
Dirección de la señal: LONG (comprar), SHORT (vender) o FLAT (neutro, sin señal).

**(inicial)** — Tipo de alerta:
- `inicial` — Primera ejecución del bot o no hay estado previo guardado. Se envía la señal tal cual.
- `cambio` — La señal cambió desde la ejecución anterior. Ej: antes era LONG, ahora SHORT.

---

## 2. Precio y Fuerza

### `Precio: $64,239.68    Fuerza: 78%`

**Precio**
Precio de cierre de la última vela al momento del análisis. Es el precio de **referencia de entrada** si se operara en ese instante.

**Fuerza: 78%**
Score compuesto (0-100%) que mide la fortaleza de la señal según el modelo:

| Rango | Interpretación |
|-------|---------------|
| 0-30% | Señal débil, probable ruido |
| 30-60% | Señal moderada |
| 60-80% | **Señal fuerte** |
| 80-100% | Señal muy fuerte |

Se calcula combinando:
- SMI Hist (momentum del mercado)
- ADX (fuerza de la tendencia)
- Score del HMM + thresholds dinámicos

---

## 3. Score LONG / SHORT

### `Score LONG: 32 | SHORT: 78  (umbral: 60)`

Los scores son números del **0 al ~100** que indican cuánta evidencia tiene el modelo para cada dirección:

- **Score SHORT: 78** → Mucha evidencia bajista. El modelo ve condiciones claras para vender.
- **Score LONG: 32** → Poca evidencia alcista. No hay condiciones de compra.
- **Umbral: 60** — Mínimo necesario para que se active una señal.

**Regla de activación:**
- Si **SHORT ≥ umbral** y **SHORT > LONG** → señal SHORT
- Si **LONG ≥ umbral** y **LONG > SHORT** → señal LONG
- Si ambos están por debajo del umbral → FLAT (sin señal)

**Ejemplos prácticos:**
- `SHORT: 78 | LONG: 32` → Victoria clara de los bajistas. Señal confiable.
- `SHORT: 62 | LONG: 58` → Casi empatados. Señal débil, posible reversión inminente.
- `SHORT: 45 | LONG: 40` → Ambos bajo el umbral. Sin señal (FLAT).

---

## 4. Régimen HMM

### `Regimen: Oso fuerte (conf 92%)`

El HMM (Hidden Markov Model) entrena 5 estados ocultos del mercado y los etiqueta automáticamente:

| Régimen | Significado |
|---------|-------------|
| Oso fuerte | Tendencia bajista consolidada |
| Oso débil | Leve presión bajista |
| Neutro / Lateral | Sin dirección clara, mercado lateral |
| Toro débil | Leve presión alcista |
| Toro fuerte | Tendencia alcista consolidada |

**conf 92%** — Confianza del régimen (0-100%):
- Mide qué tan seguro está el modelo de que el régimen asignado es el correcto.
- Se basa en la probabilidad de pertenencia al estado oculto.
- **92% es confianza muy alta**.
- Si la confianza baja de **60%**, el modelo bloquea las señales en ese timeframe para evitar falsas alertas.

---

## 5. Previo

### `Previo: LONG (hace 3 velas)`

Indica la dirección que tenía la señal en la **ejecución anterior del bot** y hace cuántas velas ocurrió el cambio.

**Ejemplo con timeline:**

| Ejecución | Hora | Score LONG | Score SHORT | Señal emitida | Previo guardado |
|-----------|------|-----------|-------------|---------------|-----------------|
| 1ª | 08:00 | 80 | 25 | **LONG** (inicial) | LONG |
| 2ª | 09:00 | 76 | 30 | *(sin cambios)* | LONG |
| 3ª | 10:00 | 32 | 78 | **SHORT** (cambio) | SHORT |
| 4ª | 11:00 | 35 | 75 | *(sin cambios)* | SHORT |

En la alerta de las 10:00: **"Previo: LONG (hace 3 velas)"** significa:
- Antes del SHORT, la señal era **LONG**.
- La señal LONG duró de 08:00 a 10:00 (3 velas en 1H, 3 días en 1D).
- Hubo un **cambio de dirección** de LONG → SHORT.

**Interpretación según el valor:**
- `Previo: LONG (hace 1 vela)` → Cambio muy brusco. Posible ruido, esperar confirmación.
- `Previo: LONG (hace 15 velas)` → Llevaba mucho tiempo en LONG. El cambio a SHORT es más significativo.
- `Previo: FLAT` → Venía de zona neutra. Las señales desde FLAT suelen ser más confiables que las reversiones.
- *(solo se muestra en alertas de tipo "cambio")*

---

## 6. Take Profit y Trailing Stop

### `TP -1.5%: $63,276.08 | Trail +1.0%: $64,882.08`

Niveles de salida calculados automáticamente según el timeframe.

**TP (Take Profit) `-1.5%: $63,276.08`**
- Take Profit fijo: si el precio alcanza ese nivel, la señal se considera ganadora.
- Para SHORT: el TP está **por debajo** del precio actual (esperas que baje).
- Para LONG: el TP está **por encima** del precio actual (`+X%`).
- El porcentaje varía por timeframe (1H usa TP más agresivo, 1W más conservador).

**Trail (Trailing Stop) `+1.0%: $64,882.08`**
- Stop dinámico que se ajusta al precio.
- Para SHORT: si el precio sube 1% desde su punto más bajo, se activa la salida.
- Para LONG: si el precio baja 1% desde su punto más alto, se activa la salida.
- Sirve para **proteger ganancias**: permite capturar la mayor parte del movimiento sin salir antes de tiempo.

**Ejemplo de trailing stop en SHORT:**
1. Entras a $64,239 (precio actual).
2. El precio baja a $63,000 → ganancia paper de $1,239.
3. El precio rebota a $64,882 → se activa el trailing stop (+1% desde $64,882).
4. Sales automáticamente con ganancia parcial, protegiendo lo ganado.

---

## 7. Expiración

### `Expira en 14 velas (~14 d)`

- La señal tiene una **ventana de validez** máxima.
- Para **1D**, 14 velas = **~14 días**.
- Si pasadas 14 velas la señal sigue activa pero no alcanzó el TP, la **expira** y se considera fallida.
- El conteo es por **velas consecutivas** con la misma señal, no por tiempo cronológico.

| Timeframe | Ventana | Significado |
|-----------|---------|-------------|
| 1H | 14 velas | ~14 horas |
| 4H | 14 velas | ~56 horas |
| 1D | 14 velas | ~14 días |
| 1W | 14 velas | ~3-4 meses |

---

## 8. Footer

### `🤖 TradingLatino HMM v4`

Identificador de la versión del bot que generó la alerta. Útil para trazabilidad y soporte.

---

## Resumen práctico

Cuando recibes esta alerta:

```
🔴 BTC-USD [1D] SHORT (inicial)
Precio: $64,239.68    Fuerza: 78%
Score LONG: 32 | SHORT: 78  (umbral: 60)
Regimen: Oso fuerte (conf 92%)
TP -1.5%: $63,276.08 | Trail +1.0%: $64,882.08
Expira en 14 velas (~14 d)
```

**Qué significa:**
1. BTC en **diario** está en régimen de **oso fuerte con 92% de confianza**.
2. El modelo generó una señal **SHORT con fuerza 78%**.
3. Los scores son concluyentes: SHORT=78 vs LONG=32, no hay competencia.
4. Si entras, el **TP está 1.5% abajo** ($63,276) y si el precio rebota 1% desde el mínimo, el **trailing stop** te saca.
5. Tienes **hasta 14 días** para que se cumpla el movimiento.

**Qué hacer:**
- Señales en **1D o 1W** → más confiables, ideales para swing trading.
- Señales en **1H o 4H** → más rápidas, para traders intradía.
- Si el score de la dirección contraria está cerca del umbral → tener precaución, posible reversión.
- Si la confianza del régimen es baja (<60%) → el modelo mismo bloquea la señal, no operar.
