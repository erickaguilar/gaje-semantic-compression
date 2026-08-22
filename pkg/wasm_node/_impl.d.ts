/* tslint:disable */
/* eslint-disable */

export class GajeWasmEngine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Chat end-to-end: recibe texto del usuario y retorna la respuesta generada.
     */
    chat(prompt: string, max_tokens: number, temperature: number, repetition_penalty: number): string;
    /**
     * Decodifica una secuencia de IDs de tokens a string.
     */
    decode(ids: Uint32Array): string;
    /**
     * Tokeniza un texto a un arreglo de IDs de tokens en JS.
     */
    encode(text: string): Uint32Array;
    /**
     * Generación completa autorregresiva en Rust nativo sobre WASM.
     */
    generate(prompt_ids: Uint32Array, max_tokens: number, temperature: number, repetition_penalty: number, stop_ids: Uint32Array): Uint32Array;
    /**
     * Retorna información arquitectónica del modelo como objeto JSON.
     */
    get_model_info(): string;
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
}
