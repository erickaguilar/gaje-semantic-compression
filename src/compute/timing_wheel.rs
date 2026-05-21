use crate::compute::event_queue::SpikeEvent;
use std::collections::VecDeque;

/// Implementación de Timing Wheel para gestión de eventos neuromórficos con costo O(1).
/// Ideal para contextos masivos donde una cola de prioridad (BinaryHeap) colapsaría.
pub struct TimingWheel {
    /// Rueda circular de slots. Cada slot contiene una lista de eventos para ese tick.
    pub slots: Vec<Vec<SpikeEvent>>,
    pub current_tick: u64,
    pub wheel_size: usize,
    /// Eventos que caen fuera del horizonte actual de la rueda (overflow).
    /// En un motor industrial completo, esto se manejaría con una segunda rueda más lenta.
    pub overflow: Vec<SpikeEvent>, 
}

impl TimingWheel {
    pub fn new(size: usize) -> Self {
        Self {
            slots: vec![Vec::new(); size],
            current_tick: 0,
            wheel_size: size,
            overflow: Vec::new(),
        }
    }

    /// Encola un nuevo evento en la rueda en tiempo O(1).
    pub fn push(&mut self, event: SpikeEvent) {
        let event_tick = event.timestamp;
        if event_tick < self.current_tick {
            // Evento pasado (no debería ocurrir en simulación causal estricta)
            return;
        }

        let horizon = self.current_tick + self.wheel_size as u64;
        if event_tick < horizon {
            let slot_idx = (event_tick % self.wheel_size as u64) as usize;
            self.slots[slot_idx].push(event);
        } else {
            // El evento está demasiado lejos en el futuro para esta rueda
            self.overflow.push(event);
        }
    }

    /// Obtiene todos los eventos programados para el tick actual.
    pub fn pop_active(&mut self) -> Vec<SpikeEvent> {
        // 1. Intentar reintegrar eventos del overflow si ahora caen dentro del horizonte
        if !self.overflow.is_empty() {
            let horizon = self.current_tick + self.wheel_size as u64;
            let mut i = 0;
            while i < self.overflow.len() {
                if self.overflow[i].timestamp < horizon {
                    let ev = self.overflow.swap_remove(i);
                    let idx = (ev.timestamp % self.wheel_size as u64) as usize;
                    self.slots[idx].push(ev);
                } else {
                    i += 1;
                }
            }
        }

        // 2. Extraer los eventos del slot actual
        let slot_idx = (self.current_tick % self.wheel_size as u64) as usize;
        std::mem::take(&mut self.slots[slot_idx])
    }

    /// Avanza el reloj de simulación.
    pub fn advance_tick(&mut self, delta: u64) {
        self.current_tick += delta;
    }

    pub fn is_empty(&self) -> bool {
        self.overflow.is_empty() && self.slots.iter().all(|s| s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_wheel_basic() {
        let mut wheel = TimingWheel::new(10);
        
        wheel.push(SpikeEvent { timestamp: 2, source_neuron_id: 1, target_layer_id: 0, target_neuron_id: 1 });
        wheel.push(SpikeEvent { timestamp: 12, source_neuron_id: 2, target_layer_id: 0, target_neuron_id: 2 }); // Overflow inicial

        // Tick 0
        assert!(wheel.pop_active().is_empty());
        
        // Tick 2
        wheel.advance_tick(2);
        let events = wheel.pop_active();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timestamp, 2);

        // Tick 12 (El evento de overflow debería haberse movido a la rueda)
        wheel.advance_tick(10);
        let events = wheel.pop_active();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timestamp, 12);
    }
}
