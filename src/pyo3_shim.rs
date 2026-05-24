#[cfg(not(feature = "python"))]
pub mod pyo3_shim {
    pub type PyResult<T> = Result<T, String>;
    #[derive(Copy, Clone)] pub struct Python<'a>(std::marker::PhantomData<&'a ()>);
    pub struct Bound<'a, T>(&'a T);
    pub struct PyModule;
    pub struct PyObject;
    impl PyObject { pub fn extract<'a, T>(&'a self, _: Python<'a>) -> Result<T, String> { Err("Not implemented".to_string()) } }
    pub struct PyRefMut<'a, T>(&'a mut T);
    pub struct PyErr;
    pub mod exceptions {
        pub struct PyValueError;
        impl PyValueError { pub fn new_err<S: ToString>(s: S) -> String { s.to_string() } }
        pub struct PyIOError;
        impl PyIOError { pub fn new_err<S: ToString>(s: S) -> String { s.to_string() } }
        pub struct PyTypeError;
        impl PyTypeError { pub fn new_err<S: ToString>(s: S) -> String { s.to_string() } }
    }
}

#[cfg(not(feature = "python"))]
pub use pyo3_shim::*;
