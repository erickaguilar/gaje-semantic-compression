/* tslint:disable */
/* eslint-disable */

export class GajeWasmEngine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Emite decisiones motoras estructuradas (Tool Calling / Actuadores).
     */
    actuate(prompt: string, tools_schema_json: string): string;
    /**
     * Ejecuta el ciclo autonómico de consolidación de memoria (sueño biológico en background).
     */
    autonomic_sleep_cycle(dedup_threshold: number): string;
    /**
     * Chat end-to-end: recibe texto del usuario y retorna la respuesta generada.
     */
    chat(prompt: string, max_tokens: number, temperature: number, repetition_penalty: number): string;
    /**
     * Chat end-to-end con inyección automática de memoria asociativa e ingesta de turno.
     */
    chat_with_memory(prompt: string, max_tokens: number, temperature: number, repetition_penalty: number, inject_rag: boolean): string;
    /**
     * Decodifica una secuencia de IDs de tokens a string.
     */
    decode(ids: Uint32Array): string;
    /**
     * Tokeniza un texto a un arreglo de IDs de tokens en JS.
     */
    encode(text: string): Uint32Array;
    /**
     * Exporta la memoria de un nicho a formato binario .gmem v2 para persistencia en IndexedDB/OPFS.
     */
    export_gmem_island(niche: string): Uint8Array;
    /**
     * Generación completa autorregresiva en Rust nativo sobre WASM.
     */
    generate(prompt_ids: Uint32Array, max_tokens: number, temperature: number, repetition_penalty: number, stop_ids: Uint32Array): Uint32Array;
    /**
     * Retorna estadísticas en tiempo real del estrato de memoria Island.
     */
    get_memory_stats(): string;
    /**
     * Retorna información arquitectónica del modelo como objeto JSON.
     */
    get_model_info(): string;
    /**
     * Importa la memoria de un nicho desde bytes binarios .gmem v2 recuperados de IndexedDB/OPFS.
     */
    import_gmem_island(niche: string, bytes: Uint8Array): void;
    /**
     * Ingesta sensorial: registra un nuevo recuerdo en el nicho de memoria Island especificado.
     */
    ingest_sensory(text: string, vector: Float32Array, niche: string, custom_id?: bigint | null): bigint;
    /**
     * Inicializa las tablas de cómputo matemático globales para WASM.
     */
    static init_engine(): void;
    /**
     * Carga el organismo genómico .flat directamente desde un ArrayBuffer / Uint8Array en JS.
     */
    static load_from_bytes(bytes: Uint8Array): GajeWasmEngine;
    /**
     * Limpia el estado interno de KV Cache para reiniciar la conversación.
     */
    reset_cache(): void;
    /**
     * Recupera los contextos más resonantes en la memoria Island como objeto JSON.
     */
    retrieve_context(query_text: string, query_vector: Float32Array, top_k: number): string;
}
