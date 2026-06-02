use crate::core::sdk::GajeSession;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// # 🏛️ Gaje-Core FFI: Interfaz de Interoperabilidad Nativa
///
/// Este módulo exporta funciones con nomenclatura C (`extern "C"`) para permitir 
/// que el motor GAJE sea utilizado desde lenguajes como C, C++, Kotlin (JNI) o Swift.

/// Carga una nueva sesión genómica.
/// Retorna un puntero opaco a `GajeSession` o NULL si falla.
#[no_mangle]
pub extern "C" fn gaje_session_load(model_path: *const c_char, memory_capacity: usize) -> *mut GajeSession {
    if model_path.is_null() { return std::ptr::null_mut(); }
    let c_str = unsafe { CStr::from_ptr(model_path) };
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match GajeSession::load(path, memory_capacity) {
        Ok(session) => Box::into_raw(Box::new(session)),
        Err(e) => {
            eprintln!("[!] FFI Error cargando modelo: {}", e);
            std::ptr::null_mut()
        },
    }
}

/// Procesa una interacción de chat.
/// Retorna una cadena C asignada en el heap de Rust que DEBE ser liberada con `gaje_string_free`.
#[no_mangle]
pub extern "C" fn gaje_session_chat(
    session: *mut GajeSession, 
    user_input: *const c_char, 
    max_tokens: usize,
    temperature: f32,
    top_p: f32
) -> *mut c_char {
    if session.is_null() || user_input.is_null() { return std::ptr::null_mut(); }
    let session = unsafe { &mut *session };
    let c_str = unsafe { CStr::from_ptr(user_input) };
    let input = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match session.chat(input, max_tokens, temperature, top_p) {
        Ok(response) => {
            match CString::new(response) {
                Ok(c_res) => c_res.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        },
        Err(e) => {
            eprintln!("[!] FFI Error en generación: {}", e);
            std::ptr::null_mut()
        },
    }
}

/// Libera la memoria de una sesión.
#[no_mangle]
pub extern "C" fn gaje_session_free(session: *mut GajeSession) {
    if !session.is_null() {
        unsafe { let _ = Box::from_raw(session); }
    }
}

/// Libera una cadena de texto retornada por el SDK.
#[no_mangle]
pub extern "C" fn gaje_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { let _ = CString::from_raw(s); }
    }
}
