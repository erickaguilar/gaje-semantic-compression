# Born Q2_0 — Archivado como research (PPL45 gibberish)

**Artefacto:** max.gaje Q2_0 v2312 99.57 MB 75 tens Llama dim256 n_blocks8 n_head4 vocab49152
**Entrenamiento:** 20e 6.6628→3.8364 ↓42.42% 8494s 88→116 tok/s PPL≈45 0 NaN/Inf +2e 3.82→3.80 +5e DNI pro3b 6.88→6.46
**Validación:** inspect Q2_0 OK, audit 0 NaN OK, GTOK 49152 encode/decode OK, throughput 150-162 tok/s, generación 0/3 gibberish (¿Quién eres? → "un mont comoio catals...", capital Germany → gibberish, largest planet → gibberish)
**Conclusión:** Viable numéricamente, no semánticamente. Capacidad dim256×8 + Q2_0 2b/peso insuficiente para 49k vocab. Requiere dim512/12L o base gaje_nano_0_5b Q4_0.
**Pipeline GPU:** ste_q2_backward, batched_gemv_q2, kl_divergence validados batch64 Vulkan RADV, 60/60 tests.
