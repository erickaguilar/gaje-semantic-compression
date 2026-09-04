#[cfg(not(feature = "python"))]
pub type PyResult<T> = Result<T, String>;

#[cfg(not(feature = "python"))]
#[derive(Copy, Clone)]
pub struct Python<'a>(std::marker::PhantomData<&'a ()>);

#[cfg(not(feature = "python"))]
impl<'a> Python<'a> {
    pub fn allow_threads<T, F>(&self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        f()
    }
}

#[cfg(not(feature = "python"))]
#[allow(dead_code)]
pub struct Bound<'a, T>(pub &'a T);

#[cfg(not(feature = "python"))]
pub struct PyModule;

#[cfg(not(feature = "python"))]
pub struct PyObject;

#[cfg(not(feature = "python"))]
impl PyObject {
    pub fn extract<'a, T>(&'a self, _: Python<'a>) -> Result<T, String> {
        Err("Not implemented".to_string())
    }
}

#[cfg(not(feature = "python"))]
#[allow(dead_code)]
pub struct PyRefMut<'a, T>(pub &'a mut T);

#[cfg(not(feature = "python"))]
pub struct PyErr;

#[cfg(not(feature = "python"))]
pub mod exceptions {
    pub struct PyValueError;
    impl PyValueError {
        pub fn new_err<S: ToString>(s: S) -> String {
            s.to_string()
        }
    }
    pub struct PyIOError;
    impl PyIOError {
        pub fn new_err<S: ToString>(s: S) -> String {
            s.to_string()
        }
    }
    pub struct PyTypeError;
    impl PyTypeError {
        pub fn new_err<S: ToString>(s: S) -> String {
            s.to_string()
        }
    }
}
