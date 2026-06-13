# 🗺️ Roadmap de Mejoras — Sistema HMM TradingLatino

## 📌 Estado General

| Mejora | Prioridad | Estado | Iniciado | Completado |
|---|---|---|---|---|
| 1. Telegram/Email alerts | 🔥 Alta | ✅ Parcial | 13-06-2026 | 13-06-2026 |
| 2. Divergencias RSI-precio | 🔥 Alta | ✅ Completado | 12-06-2026 | 12-06-2026 |
| 3. Multi-activo | 🔥 Alta | Pendiente | - | - |
| 4. Ejecución programada | 🔥 Media | Pendiente | - | - |
| 5. Métricas avanzadas en dashboard | 🔥 Media | Pendiente | - | - |
| 6. Walk-Forward Optimization | 🔥 Alta | Pendiente | - | - |
| 7. Modularización del código | 🔥 Baja | Pendiente | - | - |

---

## 1️⃣ Telegram / Email Alerts — 🔥 ALTA

**Objetivo:** Que las alertas del sistema lleguen al celular sin tener que abrir el PC.

**Implementación:**
- [x] Opción 1: Bot de Telegram (API directa con `requests`)
- [ ] Opción 2: Email con `smtplib` (viene con Python)
- [x] Elegir una opción e implementar
- [x] Probar que funcione con una alerta real

**Código estimado:** 20-30 min
**Archivos a modificar:** `tradinglatino_hmm_clean.py` (agregar función de envío)

---

## 2️⃣ Divergencias RSI-Precio — 🔥 ALTA

**Objetivo:** Detectar divergencias alcistas y bajistas entre RSI y precio para mejorar la precisión de señales.

**Implementación:**
- [ ] Implementar detección de divergencia alcista (precio mmb, RSI mmh)
- [ ] Implementar detección de divergencia bajista (precio Mmh, RSI mmb)
- [ ] Agregar como factor de peso en el scoring (+10-15 puntos)
- [ ] Verificar que no rompa los win rates actuales

**Código estimado:** 30-45 min
**Archivos a modificar:** `tradinglatino_hmm_clean.py` (funciones de señal)

---

## 3️⃣ Multi-Activo — 🔥 ALTA

**Objetivo:** Correr el análisis en varios activos simultáneamente y tener un dashboard unificado.

**Implementación:**
- [ ] Definir lista de activos a monitorear (ej. BTC, ETH, SOL, ADA)
- [ ] Loop sobre activos en `main()`
- [ ] Dashboard unificado con cards por activo
- [ ] Alertas independientes por activo

**Código estimado:** 45-60 min
**Archivos a modificar:** `tradinglatino_hmm_clean.py`, dashboard HTML

---

## 4️⃣ Ejecución Programada — 🔥 MEDIA

**Objetivo:** Que el script se ejecute automáticamente cada cierto tiempo sin intervención manual.

**Implementación:**
- [ ] Opción 1: Usar Windows Task Scheduler (instrucciones)
- [ ] Opción 2: Agregar `--daemon` mode con `time.sleep()` dentro del script
- [ ] Opción 3: Bucle con schedule (librería `schedule`)
- [ ] Decidir frecuencia (ej. cada 4h, 6h, o diario)

**Código estimado:** 10-15 min
**Archivos a modificar:** `tradinglatino_hmm_clean.py` o tarea del sistema

---

## 5️⃣ Métricas Avanzadas en Dashboard — 🔥 MEDIA

**Objetivo:** Agregar más métricas de rendimiento al dashboard HTML.

**Implementación:**
- [ ] Calcular y mostrar drawdown máximo
- [ ] Calcular y mostrar Sharpe Ratio
- [ ] Mostrar distribución de señales (% LONG vs SHORT vs FLAT)
- [ ] Agregar tabla histórica de señales recientes
- [ ] Indicador de volatilidad actual (ATR % del precio)

**Código estimado:** 30 min
**Archivos a modificar:** Funciones `_build_*` en `tradinglatino_hmm_clean.py`

---

## 6️⃣ Walk-Forward Optimization — 🔥 ALTA

**Objetivo:** Optimizar automáticamente los pesos según el régimen de mercado actual.

**Implementación:**
- [ ] Implementar ventana de entrenamiento (ej. últimos 90 días)
- [ ] Buscar combinación óptima de pesos en ventana de entrenamiento
- [ ] Probar en ventana de test (siguientes 30 días)
- [ ] Detectar automáticamente cambios de régimen y re-optimizar

**Código estimado:** 1-2 horas
**Archivos a modificar:** `tradinglatino_hmm_clean.py` (nuevas funciones)

---

## 7️⃣ Modularización del Código — 🔥 BAJA

**Objetivo:** Separar el monstruo de ~5000 líneas en módulos manejables.

**Implementación:**
- [ ] Crear estructura de carpetas `tradinglatino/`
- [ ] Separar `indicators.py` (indicadores técnicos)
- [ ] Separar `signals.py` (lógica de señales)
- [ ] Separar `hmm_model.py` (detectores de régimen)
- [ ] Separar `alerts.py` (sistema de alertas + notificaciones)
- [ ] Separar `dashboard.py` (generación HTML)
- [ ] Separar `backtest.py` (backtesting)
- [ ] Separar `config.py` (constantes, pesos, thresholds)
- [ ] Verificar que todo sigue funcionando igual

**Código estimado:** 45 min
**Archivos a modificar:** Todos — reestructuración completa

---

## 📝 Notas de Sesiones

<!-- Aquí iremos anotando qué se hizo en cada sesión -->

| Fecha | Sesión | Qué se hizo |
|---|---|---|
| 12-06-2026 | Implementación RSI Divergencias | Detecta divergencias alcistas/bajistas RSI-Precio con swing points. Peso base W_RSI_DIVERGENCE=12 + bonus 4pts si RSI extremo. Integrado en score compuesto LONG/SHORT y dashboard breakdown. |

---

## 🧠 Cómo Usar Este Archivo

1. Abre Codebuff y di: *"Lee el ROADBOARD.md y dime las mejoras pendientes"*
2. Codebuff leerá el archivo y sabrá exactamente qué falta
3. Pide la mejora que quieras implementar: *"Implementa la mejora #1, Telegram alerts"*
4. Después de cada implementación, se actualizará el ROADBOARD.md automáticamente

---

*Última actualización: Junio 2026*
