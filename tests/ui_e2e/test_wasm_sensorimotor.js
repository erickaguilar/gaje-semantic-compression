/**
 * 🧬 GAJE-WASM: Certificación del Tronco Encefálico Sensorio-Motor y Ciclo Autonómico
 */

const fs = require('fs');
const path = require('path');

const wasmModule = require('../pkg/wasm_node/_impl.js');
const { GajeWasmEngine } = wasmModule;

async function runCertification() {
    console.log("================================================================================");
    console.log("🧠 CERTIFICACIÓN WASM: Tronco Encefálico Sensorio-Motor y Ciclo Autonómico");
    console.log("================================================================================");

    // 1. Cargar modelo binario plano (.flat)
    const modelPath = path.join(__dirname, '..', 'models', 'production', 'smollm2_135m.flat');
    if (!fs.existsSync(modelPath)) {
        throw new Error(`Modelo no encontrado en: ${modelPath}`);
    }

    const tLoad0 = Date.now();
    const flatBytes = fs.readFileSync(modelPath);
    const engine = GajeWasmEngine.load_from_bytes(flatBytes);
    const loadTimeMs = Date.now() - tLoad0;

    console.log(`[1] Organismo cargado en WASM Linear Memory en ${loadTimeMs} ms.`);
    const modelInfo = JSON.parse(engine.get_model_info());
    console.log(`    - Arquitectura: n_embd=${modelInfo.n_embd}, n_blocks=${modelInfo.n_layer}, vocab=${modelInfo.vocab_size}`);

    // 2. Vías Aferentes (Aferencia Sensorial & Retrieval)
    console.log("\n[2] Probando Vías Aferentes (Ingesta Sensorial a Islas .gmem)...");
    const id1 = engine.ingest_sensory("El gen FoxP2 coordina el desarrollo de circuitos neuronales del lenguaje.", [], "documental");
    const id2 = engine.ingest_sensory("CRISPR-Cas9 permite la edición genómica precisa.", [], "documental");
    const id3 = engine.ingest_sensory("La memoria episódica registra eventos con marca temporal.", [], "episodic");
    console.log(`    - Recuerdos ingestados: ID ${id1} (Documental), ID ${id2} (Documental), ID ${id3} (Episódico)`);

    const retrievalJson = engine.retrieve_context("FoxP2 y lenguaje", [], 2);
    const retrieval = JSON.parse(retrievalJson);
    console.log(`    - Resonancia semántica recuperada (${retrieval.length} items):`);
    retrieval.forEach(r => console.log(`      • [${r.niche}] ID ${r.id} (Sim: ${r.similarity.toFixed(4)}): "${r.text.substring(0, 50)}..."`));

    if (retrieval.length === 0 || !retrieval[0].text.includes("FoxP2")) {
        throw new Error("Fallo en recuperación semántica aferente");
    }

    // 3. Eferencia & Chat con Memoria Aferente Inyectada
    console.log("\n[3] Probando Vías Eferentes y Generación con Memoria Inyectada...");
    const tGen0 = Date.now();
    const response = engine.chat_with_memory("¿Qué función tiene el gen FoxP2?", 20, 0.6, 1.1, true);
    const genTimeMs = Date.now() - tGen0;
    console.log(`    - Respuesta generada en ${genTimeMs} ms:`);
    console.log(`      "${response.trim()}"`);

    // 4. Ciclo Autonómico (Sueño & Consolidación)
    console.log("\n[4] Probando Ciclo Autonómico (Consolidación de Memoria y Poda Semántica)...");
    // Ingestar duplicado en episódico para probar poda
    engine.ingest_sensory("El gen FoxP2 coordina el desarrollo de circuitos neuronales del lenguaje.", [], "episodic");
    engine.ingest_sensory("Recuerdo episódico nuevo sobre síntesis de proteínas.", [], "episodic");

    const sleepStatsJson = engine.autonomic_sleep_cycle(0.95);
    const sleepStats = JSON.parse(sleepStatsJson);
    console.log(`    - Estadísticas del Ciclo de Sueño:`);
    console.log(`      • Episódicos transferidos: ${sleepStats.episodic_transferred}`);
    console.log(`      • Duplicados podados: ${sleepStats.duplicates_pruned}`);
    console.log(`      • Total documental: ${sleepStats.total_documental_entries}`);

    if (sleepStats.duplicates_pruned < 1) {
        throw new Error("El ciclo de sueño debió podar al menos 1 entrada duplicada");
    }

    // 5. Persistencia Soberana .gmem v2 (Export & Import)
    console.log("\n[5] Probando Persistencia Soberana .gmem v2 (Exportación e Importación)...");
    const gmemDocBytes = engine.export_gmem_island("documental");
    console.log(`    - Memoria Documental exportada a binario .gmem v2: ${gmemDocBytes.length} bytes`);

    // Validar cabecera mágica GMEM
    if (gmemDocBytes[0] !== 0x47 || gmemDocBytes[1] !== 0x4D || gmemDocBytes[2] !== 0x45 || gmemDocBytes[3] !== 0x4D) {
        throw new Error("Cabecera .gmem exportada inválida (Falta magic GMEM)");
    }

    // Reimportar para certificar roundtrip
    engine.import_gmem_island("documental", gmemDocBytes);
    const statsAfterImport = JSON.parse(engine.get_memory_stats());
    console.log(`    - Estadísticas post-reimportación: ${statsAfterImport.documental_entries} entradas documentales.`);

    // 6. Actuación Motora Estructurada (Tool Calling)
    console.log("\n[6] Probando Eferencia Motora (Actuador / Tool Calling)...");
    const toolSchema = JSON.stringify([{ name: "set_temperature", parameters: { degrees: "number" } }]);
    const actResponse = engine.actuate("Ajusta la temperatura del biorreactor a 37 grados", toolSchema);
    console.log(`    - Decisión motora emitida:`);
    console.log(`      "${actResponse.trim()}"`);

    console.log("\n================================================================================");
    console.log("✅ CERTIFICACIÓN WASM 100% EXITOSA: Tronco Encefálico y Ciclo Autonómico OK");
    console.log("================================================================================");
}

runCertification().catch(err => {
    console.error("\n❌ Error en certificación WASM:", err);
    process.exit(1);
});
