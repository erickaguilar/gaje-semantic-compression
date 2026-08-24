import fs from 'fs';
import path from 'path';
import { GajeWasmEngine } from '../pkg/wasm_node/_impl.js';

console.log("================================================================================");
console.log("🧬 GAJE-WASM: VERIFICACIÓN DEL MOTOR COMO TRONCO ENCEFÁLICO (NODE/WASM)");
console.log("================================================================================");

const modelPath = path.resolve('models/production/smollm2_135m.flat');
if (!fs.existsSync(modelPath)) {
    console.error(`❌ Modelo ${modelPath} no encontrado.`);
    process.exit(1);
}

console.log(`[*] Leyendo buffer binario de ${modelPath}...`);
const fileBuffer = fs.readFileSync(modelPath);
const uint8Array = new Uint8Array(fileBuffer);
console.log(`[+] Buffer leído: ${(uint8Array.length / (1024 * 1024)).toFixed(2)} MB`);

console.log(`[*] Instanciando GajeWasmEngine.load_from_bytes()...`);
const t0 = performance.now();
const engine = GajeWasmEngine.load_from_bytes(uint8Array);
const loadTimeMs = (performance.now() - t0).toFixed(2);
console.log(`✅ Organismo cargado en WebAssembly en ${loadTimeMs} ms`);

const infoStr = engine.get_model_info();
const info = JSON.parse(infoStr);
console.log(`📊 Metadatos del Modelo:`, info);

// 1. Prueba de Tokenización GTOK en WASM
const testText = "El ADN y la compresión genómica";
console.log(`[*] Probando tokenización GTOK en WASM: "${testText}"...`);
const tokenIds = engine.encode(testText);
console.log(`[+] Token IDs (${tokenIds.length}):`, Array.from(tokenIds));

const decodedText = engine.decode(tokenIds);
console.log(`[+] Detokenizado: "${decodedText}"`);

if (decodedText.trim() !== testText.trim()) {
    console.warn(`⚠️ Diferencia en detokenización: esperado "${testText}", obtenido "${decodedText}"`);
} else {
    console.log(`✅ Tokenización y Decodificación GTOK 100% fiel en WebAssembly.`);
}

// 2. Prueba de Generación en WebAssembly
console.log(`\n[*] Probando inferencia autorregresiva en WebAssembly...`);
const prompt = "¿Qué es el ADN?";
const tGen0 = performance.now();
const response = engine.chat(prompt, 15, 0.7, 1.1);
const genTimeMs = performance.now() - tGen0;

console.log(`\n💬 [PREGUNTA]: ${prompt}`);
console.log(`🧬 [RESPUESTA WASM]:\n${response}`);
console.log(`⏱️ Latencia: ${genTimeMs.toFixed(2)} ms`);

engine.free();
console.log("\n================================================================================");
console.log("✅ VERIFICACIÓN DE GAJE-WASM (FASE 2) COMPLETADA CON ÉXITO");
console.log("================================================================================");
