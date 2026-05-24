#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::compute::event_queue::SpikeEvent;
use std::collections::BinaryHeap;

#[cfg_attr(feature = "python", pyclass)]
pub struct NeuromorphicScheduler {
    pub event_queue: BinaryHeap<SpikeEvent>,
    pub current_time: f32,
    pub max_time: f32,
}

impl NeuromorphicScheduler {
    pub fn new(max_time: f32) -> Self {
        NeuromorphicScheduler {
            event_queue: BinaryHeap::new(),
            current_time: 0.0,
            max_time,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl NeuromorphicScheduler {
    #[new]
    pub fn py_new(max_time: f32) -> Self {
        Self::new(max_time)
    }

    pub fn schedule_spike(&mut self, timestamp: f32, neuron_idx: usize, layer_idx: usize) {
        if timestamp <= self.max_time {
            self.event_queue.push(SpikeEvent { timestamp, neuron_idx, layer_idx });
        }
    }
}
