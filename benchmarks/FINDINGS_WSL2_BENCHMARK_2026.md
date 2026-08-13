# Benchmark GAJE `.flat` en WSL2 — Hallazgos (2026-08-11)

**Objetivo:** reproducir el rendimiento certificado del motor GAJE con
`Qwen2.5-1.5B` cuantizado híbrido (`qwen2_5_1_5b_q4_0.gaje.flat`, embeddings
FP32 + cuerpo Q4_0) según README (`11.3–12.1 tok/s`, cold-start `<0.75 ms`).

**Entorno de ejecución:**
- WSL2 (VM) sobre Windows 11.
- CPU: Intel i7-1370P (20 hilos expuestos en guest; compartidos con host).
- RAM VM: 7 GB totales + 2 GB SWAP.
- Modelo (2.7 GB) residiendo en unidad Windows `E:` => montado en `/mnt/e`
  (filesystem cruzando el puente de traducción 9p de WSL2).
- Motor Rust compilado localmente (`_impl.abi3.so`), mismos `.venv-linux`.

---

## Resultados medidos

| Métrica | Resultado | Claim README (AMD 5800H) | Δ |
| :--- | :---: | :---: | :---: |
| Cold-start mmap (modelo en `/mnt/e`, NTFS/9p) | **~42 s** | <0.75 ms | ✗ |
| Cold-start mmap (modelo copiado a ext4 nativo) | **~2.5 s** | <0.75 ms | ✗ |
| Prefill (13 tok, prompt) | **1.4–1.8 tok/s** | — | — |
| Decode (64 tok, greedy) | **1.3–1.55 tok/s** | 11.3–12.1 tok/s | ✗ (~8x) |

---

## Diagnóstico

### 1. El cold-start mmap NO es `<0.75 ms`, y el culpable es el filesystem Windows
El `mmap` de GAJE es **perezoso**: tras "cargar" el modelo, `mRSS` permanece en
`0 kB`. La lectura física real ocurre al primer *forward* (page-fault).

- Con el `.flat` en `/mnt/e` (NTFS vía 9p), el arranque tardó **~42 s**.
- Copiado el mismo archivo a **ext4 nativo** bajó a **~2.5 s**.
- Copiar el archivo de 2.7 GB Windows→ext4 a través del puente 9p tardó 40 s.

→ El número `<0.75 ms` captura solo el `mmap`+parseo de cabecera **cuando las
páginas están en RAM física servible**. En cualquier filesystem donde los
page-faults son costosos (9p, red, disco frío) domina el materializado real.

### 2. El throughput sostenido NO alcanza los claims (1.3–1.5 vs 11–12 tok/s)
La pasada en caliente (páginas ya mapeadas) sigue dando ~1.3–1.5 tok/s, no un
transient de page-fault. Esto es ≈8x por debajo del README. Factores imparciales
de este entorno:

- VM WSL2 con solo 7 GB de RAM compartida (+2 GB SWAP) => el mmap de 2.7 GB
  puede convoy contra el host y el disco, induciendo thrash.
- CPU virtual compartida, sin control exclusivo de los 20 hilos.
- El build no está optimizado para el host real del benchmark de referencia.

---

## Verdades parciales vs. medidas objetivas

- ✅ **Mappabilidad / formato `.flat`**: el mmap perezoso funciona; RSS real
  empieza en 0 y se materializa bajo demanda. En Linux nativo con RAM suficiente
  es plausible alcanzar sub-ms de *carga*, situando la cifra `<0.75 ms` como
  factible en ese contexto.
- ❌ **Reproducibilidad a ciegas**: este entorno **no es representativo** y no
  debe tomarse como evidencia en contra de los claims del README, ni a favor:
  simplemente es un entorno (VM + filesystem Windows) donde el motor no puede
  desplegar su rendimiento.

---

## Conclusiones para los autores

1. **Documentar la dependencia del filesystem.** El cold-start de `mmap` solo es
   submilisegundo con páginas residentes/ext4 caliente. En `/mnt/*` (WSL) la
   cifra se va a decenas de segundos. Recomendable advertirlo en README.
2. **Reproducir el benchmark en hardware nativo Linux** (ext4, ≥8 GB RAM) para
   validar 11+ tok/s. Considerar ofrecer `benchmarks/performance/gaje_flat_benchmark.py`.
3. **Comparar contra llama.cpp/GGUF** en la misma máquina para aislar la ventaja
   real del formato `.flat` frente a la cuantización Q4_0 estándar.
4. **Consistencia de metadatos**: `Cargo.toml` declara `dna-semantic-compression
   v1.0.0-alpha` mientras README habla de GAJE v1.6.0-alpha; sincronizar.

---

## Cómo reejecutar

```bash
# En WSL2, primero copia el modelo a ext4 (evitar /mnt/* y el puente 9p)
cp models/production/qwen2_5_1_5b_q4_0.gaje.flat ~/qwen15_flat.gaje.flat

# Desde la raíz del repo, con el venv que tenga _impl compilado
.venv-linux/bin/python benchmarks/performance/gaje_flat_benchmark.py \
    --model ~/qwen15_flat.gaje.flat \
    --tokenizer temp_tokenizer/tokenizer.json \
    --tokens 64
```