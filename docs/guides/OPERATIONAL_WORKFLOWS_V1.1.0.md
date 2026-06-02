# 🛠️ Guía de Flujos Operativos: Protocolo GAJE v1.1.0

Esta guía documenta los comandos y procedimientos estándar utilizados para el desarrollo, validación y nacimiento de modelos genómicos en el ecosistema GAJE.

---

## 1. Ciclo de Construcción y Soberanía Rust

Para asegurar que el motor de alto rendimiento esté sincronizado con la capa de investigación en Python.

### Compilación Completa (Release)
Utilizar este comando después de modificar archivos en `src/` o `src/compute/`.
```bash
maturin build --release --features python
```

### Instalación de la Extensión
Obligatorio para que los scripts de Python reconozcan las nuevas funciones nativas (como el analizador de entropía).
```bash
pip install target/wheels/dna_semantic_compression-*-cp310-abi3-*.whl --no-deps --force-reinstall
```

---

## 2. Validación de Integridad

Protocolo de tres niveles para garantizar la estabilidad del organismo genómico.

### Nivel 1: Tests Unitarios (Rust)
Valida la física (Euler-Lagrange) y la aritmética toroidal.
```bash
cargo test nn::spiking::neuron
cargo test nn::spiking::layer
```

### Nivel 2: Integración (Python/Rust)
Asegura que la comunicación entre capas no tenga regresiones.
```bash
pytest tests/integration/test_integration_v060.py
```

### Nivel 3: Inferencia en Vivo
Prueba de coherencia generativa con el modelo Silver Adult.
```bash
python examples/core_demos/chat_genomico.py --model models/silver_adult_anchored.gaje --prompt "Hola" --tokens 50
```

---

## 3. Nacimiento de Modelos desde Cero (Born-Genomic)

Flujo para engendrar nuevos micro-organismos optimizados quirúrgicamente.

### Ejecución de la Secuencia de Nacimiento
```bash
python scripts/research/birth_micro_organism.py
```

### Acciones del Flujo:
1.  **Mapeo de Entropía:** Escaneo de dimensiones críticas mediante Shannon.
2.  **Inyección de Anclas:** Protección F16 automática para el Top informativo.
3.  **Calibración Física:** Configuración del motor de mínima acción.
4.  **Persistencia:** Almacenamiento en formato binario `.gaje`.

---

## 4. Gestión de Energía y Rendimiento

Comandos para optimizar la ejecución en hardware móvil (ARM big.LITTLE).

### Verificación de Perfil de Energía
```bash
cargo run --release --bin gaje-power-demo
```

---
*Documento mantenido bajo el Protocolo GAJE-Flow v1.1.0*
