# PLAN DE IMPLEMENTACION: Markov Switching como Alternativa al HMM

## Fecha: 10 de Junio, 2026

========================================================================
INDICE
========================================================================
1. PROBLEMA ACTUAL
2. OBJETIVO
3. QUE ES MARKOV SWITCHING?
4. ARCHIVOS A CREAR
5. PLAN DE IMPLEMENTACION
6. CODIGO A IMPLEMENTAR
7. COMO EJECUTAR
8. METRICAS DE EXITO

========================================================================
1. PROBLEMA ACTUAL
========================================================================

HMM actual: tasa de deteccion ~40% (63/158 cambios de senal en 1d)
Win Rate de senales: ~82-96%

Diferencia: 42-56 puntos porcentuales de gap.
El HMM no anticipa bien los cambios de senal porque solo contribuye
8 puntos (W_HMM_REGIME=8) al score compuesto de ~85+ puntos totales.

========================================================================
2. OBJETIVO
========================================================================

Implementar Markov Switching Autoregression (statsmodels) como alternativay comparar resultados lado a lado.

========================================================================
3. QUE ES MARKOV SWITCHING?
========================================================================

Modelo econometrico de Hamilton (1989) para series de tiempo con
cambios de regimen. Ventajas sobre HMM:
- Significancia estadistica (p-values, std errors)
- Mas conservador (menos falsos positivos)
- Ya instalado (statsmodels 0.14.6)

Desventajas:
- Solo usa returns (no 10+ features como HMM)
- Solo 2 estados (vs 3-5 del HMM)

========================================================================
4. ARCHIVOS A CREAR
========================================================================

A) tradinglatino_regime_switching.py (NUEVO)
   - Modulo con funciones para Markov Switching
   - Dependencias: statsmodels, pandas, numpy
   - Funciones:
     * fit_markov_switching() - ajusta modelo MS
     * align_ms_states_to_df() - alinea estados al DataFrame
     * classify_ms_state() - clasifica bullish/bearish/neutral
     * describe_ms_state() - descripcion legible del estado
     * compute_ms_state_summary() - resumen de estados
     * find_ms_regime_changes() - detecta cambios de regimen

B) comparacion_modelos.py (NUEVO)
   - Script que ejecuta HMM y MS sobre mismos datos
   - Compara resultados lado a lado
   - Genera reporte HTML: comparacion_modelos_BTC-USD.html
   - Importa desde:
     * tradinglatino_hmm_clean.py (indicadores, HMM, carga datos)
     * tradinglatino_regime_switching.py (MS)
     * simulacion_alertas_tendencia.py (cross_reference, signal_changes)

========================================================================
5. PLAN DE IMPLEMENTACION PASO A PASO
========================================================================

PASO 1: Crear tradinglatino_regime_switching.py
PASO 2: Crear comparacion_modelos.py
PASO 3: Ejecutar: cd C:\FreeBuff && python comparacion_modelos.py
PASO 4: Analizar reporte HTML generado
PASO 5: Si MS es superior, integrar en el dashboard

========================================================================
6. CODIGO A IMPLEMENTAR
========================================================================

VER LA CONVERSACION COMPLETA DE CODEBUFF DEL 10-06-2026
para el codigo completo de ambos archivos.

Resumen de funciones clave:

from statsmodels.tsa.regime_switching import MarkovAutoregression
model = MarkovAutoregression(returns, k_regimes=2, order=1)
result = model.fit()
states = result.smoothed_marginal_probabilities.idxmax(axis=1)

========================================================================
7. COMO EJECUTAR
========================================================================

cd C:\FreeBuff
C:\Users\Carlos Mundaray\AppData\Local\Programs\Python\Python312\python comparacion_modelos.py

Genera: comparacion_modelos_BTC-USD.html

========================================================================
8. METRICAS DE EXITO
========================================================================

MS es mejor si:
- Tasa deteccion > 50% (vs ~40% HMM)
- Falsos positivos < 100 (vs ~176 HMM)
- Falsos negativos < 80 (vs ~95 HMM)
- Cambios mas estables y significativos
