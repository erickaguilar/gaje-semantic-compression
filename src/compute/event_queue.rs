#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Representa un evento de disparo (Spike) en el tiempo con precisión de fase e intensidad.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeEvent {
    pub timestamp: u64,
    pub phase_offset: u8,
    pub intensity: f32,
    pub source_neuron_id: usize,
    pub target_layer_id: usize,
    pub target_neuron_id: usize,
}

#[cfg_attr(feature = "python", pymethods)]
impl SpikeEvent {
    #[cfg(feature = "python")]
    #[new]
    pub fn new_py(timestamp: u64, phase_offset: u8, intensity: f32, source_neuron_id: usize, target_layer_id: usize, target_neuron_id: usize) -> Self {
        Self { timestamp, phase_offset, intensity, source_neuron_id, target_layer_id, target_neuron_id }
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_timestamp(&self) -> u64 { self.timestamp }
    #[cfg(feature = "python")]
    #[getter]
    pub fn get_phase_offset(&self) -> u8 { self.phase_offset }
    #[cfg(feature = "python")]
    #[getter]
    pub fn get_intensity(&self) -> f32 { self.intensity }
    #[cfg(feature = "python")]
    #[getter]
    pub fn get_source_neuron_id(&self) -> usize { self.source_neuron_id }
    #[cfg(feature = "python")]
    #[getter]
    pub fn get_target_layer_id(&self) -> usize { self.target_layer_id }
    #[cfg(feature = "python")]
    #[getter]
    pub fn get_target_neuron_id(&self) -> usize { self.target_neuron_id }
}

impl Eq for SpikeEvent {}
impl PartialOrd for SpikeEvent { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        let res = other.timestamp.cmp(&self.timestamp);
        if res == Ordering::Equal { other.phase_offset.cmp(&self.phase_offset) } else { res }
    }
}

pub struct EventQueue {
    queue: BinaryHeap<SpikeEvent>,
    pub current_tick: u64,
}
impl Default for EventQueue { fn default() -> Self { Self::new() } }
impl EventQueue {
    pub fn new() -> Self { Self { queue: BinaryHeap::new(), current_tick: 0 } }
    pub fn push(&mut self, event: SpikeEvent) { self.queue.push(event); }
    pub fn pop_active(&mut self) -> Option<SpikeEvent> {
        if let Some(event) = self.queue.peek() { if event.timestamp <= self.current_tick { return self.queue.pop(); } }
        None
    }
    pub fn advance_tick(&mut self, delta: u64) { self.current_tick += delta; }
    pub fn skip_to_next_event(&mut self) { if let Some(event) = self.queue.peek() { if event.timestamp > self.current_tick { self.current_tick = event.timestamp; } } }
    pub fn is_empty(&self) -> bool { self.queue.is_empty() }
    pub fn len(&self) -> usize { self.queue.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_event_queue_ordering() {
        let mut eq = EventQueue::new();
        eq.push(SpikeEvent { timestamp: 10, phase_offset: 0, intensity: 1.0, source_neuron_id: 1, target_layer_id: 0, target_neuron_id: 1 });
        eq.push(SpikeEvent { timestamp: 5, phase_offset: 5, intensity: 1.0, source_neuron_id: 2, target_layer_id: 0, target_neuron_id: 2 });
        eq.push(SpikeEvent { timestamp: 5, phase_offset: 2, intensity: 1.0, source_neuron_id: 4, target_layer_id: 0, target_neuron_id: 4 });
        eq.advance_tick(5);
        let ev1 = eq.pop_active().unwrap();
        assert_eq!(ev1.timestamp, 5); assert_eq!(ev1.phase_offset, 2);
    }
}
