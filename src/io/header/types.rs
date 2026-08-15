// =============================================================================
// types — QuantFormat y HeaderError
// =============================================================================
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    LegacyCentroids = 0,
    Q4_0 = 1,
    Q8_0 = 2,
    Unknown,
}

#[derive(Debug)]
pub enum HeaderError {
    TooShort,
    InvalidMagic,
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "Header buffer is less than 4096 bytes"),
            Self::InvalidMagic => write!(f, "Invalid magic bytes (expected 'GAJE')"),
        }
    }
}

impl std::error::Error for HeaderError {}
