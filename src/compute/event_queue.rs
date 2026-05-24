#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::cmp::Ordering;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct SpikeEvent {
    pub timestamp: f32,
    pub neuron_idx: usize,
    pub layer_idx: usize,
}

#[cfg_attr(feature = "python", pymethods)]
impl SpikeEvent {
    #[cfg(feature = "python")]
    #[new]
    pub fn new(timestamp: f32, neuron_idx: usize, layer_idx: usize) -> Self {
        SpikeEvent { timestamp, neuron_idx, layer_idx }
    }
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool { self.timestamp == other.timestamp }
}

impl Eq for SpikeEvent {}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering { other.timestamp.partial_cmp(&self.timestamp).unwrap_or(Ordering::Equal) }
}
