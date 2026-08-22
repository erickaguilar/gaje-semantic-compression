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

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly gaje_session_chat: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly gaje_session_free: (a: number) => void;
    readonly gaje_session_load: (a: number, b: number) => number;
    readonly gaje_string_free: (a: number) => void;
    readonly __wbg_gajewasmengine_free: (a: number, b: number) => void;
    readonly gajewasmengine_chat: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly gajewasmengine_decode: (a: number, b: number, c: number, d: number) => void;
    readonly gajewasmengine_encode: (a: number, b: number, c: number, d: number) => void;
    readonly gajewasmengine_generate: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly gajewasmengine_get_model_info: (a: number, b: number) => void;
    readonly gajewasmengine_init_engine: () => void;
    readonly gajewasmengine_load_from_bytes: (a: number, b: number, c: number) => void;
    readonly gajewasmengine_reset_cache: (a: number) => void;
    readonly __wbindgen_export: (a: number) => void;
    readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export3: (a: number, b: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
