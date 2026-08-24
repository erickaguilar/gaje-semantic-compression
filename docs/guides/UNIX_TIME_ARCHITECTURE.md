# ⏱️ Arquitectura de Tiempo Unix (POSIX Epoch) en GAJE

**Fecha de Implementación:** 23 de Agosto de 2026
**Módulos Afectados:** `server.py` (Backend Linux), `GajeHelixDB` (`storage.js`), `composer.js`, `utils.js`, `engine.js`
**Estado:** ✅ Aprobado y Certificado en Producción

---

## 1. Contexto y Justificación del Cambio

Anteriormente, el registro temporal de los turnos de conversación se formateaba exclusivamente en cadenas de texto locales arbitrarias (ej. `23:14:08::412`). Aunque legibles para el ojo humano, este formato presentaba limitaciones críticas para un sistema de inteligencia artificial y memoria episódica soberana (`.gmem` / `IndexedDB`):

1. **Ambigüedad de Husos Horarios y Desfases Geográficos:** Una cadena `"17:30:00"` no contiene información sobre la zona horaria (UTC-6, UTC+1, etc.). Si un usuario viaja, cambia de huso o exporta su base de datos a un servidor remoto, la cronología colapsa.
2. **Costo Computacional de Ordenamiento:** Ordenar cadenas de texto o re-parsear fechas con `new Date(str)` en colecciones masivas de `IndexedDB` introduce sobrecarga de CPU y riesgo de errores de localización.
3. **Pérdida de Precisión Sub-milisegundo:** La inferencia en kernels nativos de Rust y streaming SSE opera a escala de microsegundos ($\mu s$). Guardar únicamente minutos y segundos truncaba la telemetría de rendimiento.

Para solucionar estos problemas de raíz, **GAJE adoptó el Tiempo Unix (POSIX Epoch) como el formato canónico universal de almacenamiento e intercambio de datos**.

---

## 2. Fundamentos del Tiempo Unix (POSIX Time)

El **Tiempo Unix** representa el número absoluto de segundos transcurridos desde el **1 de enero de 1970 a las 00:00:00 UTC** (*Unix Epoch*):

$$t_{\text{posix}} \in \mathbb{R}^+, \quad t_{\text{posix}} = \text{segundos desde Epoch}$$

### Propiedades Clave:
* **Invarianza Espacial:** Es exactamente el mismo número en cualquier punto de la Tierra y en cualquier sistema operativo (Linux, macOS, Windows, Android).
* **Comparabilidad Monotónica $O(1)$:** Dos eventos $A$ y $B$ se ordenan simplemente con $t_A < t_B$.
* **Precisión de Punto Flotante:** Almacenado como número real (ej. `1771891234.567890`), preserva microsegundos de inferencia.

---

## 3. Especificación Técnica de la Implementación

```
┌────────────────────────────────────────────────────────┐
│               Kernel Linux (server.py)                │
│    time.time() -> 1771891234.567 (Unix POSIX)          │
│    time.monotonic() -> Inferencia inmune a saltos NTP  │
└──────────────────────────┬─────────────────────────────┘
                           │ SSE Stream / JSON Payload
                           ▼
┌────────────────────────────────────────────────────────┐
│            Capa de Persistencia (IndexedDB)            │
│    GajeHelixDB.messages -> timestampPosix: Float       │
│    Índice B-Tree indexado numéricamente                │
└──────────────────────────┬─────────────────────────────┘
                           │ Renderizado Reactivo
                           ▼
┌────────────────────────────────────────────────────────┐
│          Interfaz de Usuario (HTML5 + CSS)             │
│  <time datetime="2026-08-23T23:22:06.412Z"             │
│        data-unix="1771891234.567"                      │
│        data-tooltip="POSIX: 1771891234.567s">          │
│    23:22:06::412  <!-- Máscara visual de alta fidelidad-->│
│  </time>                                               │
└────────────────────────────────────────────────────────┘
```

### A. Emisión en Backend (`server.py`)
Tanto en `/api/chat` como en `/api/chat/stream`, el backend emite:
```python
"timestamp_posix": time.time(),
"server_time": datetime.now().strftime("%H:%M:%S::%f")[:-3]
```

### B. Persistencia en IndexedDB (`GajeHelixDB`)
Cada registro guardado en la base de datos almacena:
* `timestampPosix`: Marca numérica flotante de Unix.
* `savedAt`: Milisegundos de persistencia local (`Date.now()`).
* `meta.timestamp_posix`: Metadatos de telemetría inmutables.

### C. Conversión y Renderizado en Frontend (`utils.js` & `composer.js`)
* La hora visual visible (`23:22:06::412`) se computa dinámicamente como una máscara de presentación mediante `ChatUtils.formatExactTime(posixVal)`.
* El elemento `<time>` expone el estándar W3C ISO 8601 (`datetime`) y el valor POSIX exacto en atributos de datos (`data-unix`).

---

## 4. Matriz de Beneficios

| Dimensión | Antes (String Local) | Ahora (Unix POSIX) |
| :--- | :--- | :--- |
| **Universalidad** | Dependiente de la zona horaria del cliente | Universal e independiente de ubicación |
| **Búsqueda e Índices** | Escaneo secuencial / conversiones lentas | Búsqueda indexada $O(\log n)$ en IndexedDB |
| **Interoperabilidad** | Difícil de exportar e integrar con Rust | 100% interoperable (`SystemTime` en Rust/Python) |
| **Auditoría e Islas .gmem** | Posible colisión en turnos rápidos | Cronología estricta a nivel de microsegundo |

---

## 5. Conclusión

La adopción del Tiempo Unix garantiza que GAJE sea un sistema de memoria episódica semántica distribuida, matemáticamente robusto, transferible y preparado para entornos multi-agente en tiempo real.
