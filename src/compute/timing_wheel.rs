use crate::compute::event_queue::SpikeEvent;

/// Implementación de Timing Wheel para gestión de eventos neuromórficos con costo O(1) y precisión de fase.
/// Cada tick lógico se divide en 16 sub-ticks (Phase Coding).
pub struct TimingWheel {
    /// Rueda circular de slots. Cada slot contiene 16 vectores (uno por cada sub-tick de fase).
    pub slots: Vec<[Vec<SpikeEvent>; 16]>,
    pub current_tick: u64,
    pub wheel_size: usize,
    /// Eventos que caen fuera del horizonte actual de la rueda (overflow).
    pub overflow: Vec<SpikeEvent>, 
}

impl TimingWheel {
    pub fn new(size: usize) -> Self {
        let mut slots = Vec::with_capacity(size);
        for _ in 0..size {
            // Inicializar cada slot con 16 sub-vectores vacíos
            let sub_slots: [Vec<SpikeEvent>; 16] = Default::default();
            slots.push(sub_slots);
        }
        Self {
            slots,
            current_tick: 0,
            wheel_size: size,
            overflow: Vec::new(),
        }
    }

    /// Encola un nuevo evento en la rueda en tiempo O(1).
    pub fn push(&mut self, event: SpikeEvent) {
        let event_tick = event.timestamp;
        if event_tick < self.current_tick {
            return;
        }

        let horizon = self.current_tick + self.wheel_size as u64;
        if event_tick < horizon {
            let slot_idx = (event_tick % self.wheel_size as u64) as usize;
            let phase_idx = (event.phase_offset & 0x0F) as usize;
            self.slots[slot_idx][phase_idx].push(event);
        } else {
            self.overflow.push(event);
        }
    }

    /// Obtiene todos los eventos programados para el tick actual, ordenados por fase.
    pub fn pop_active(&mut self) -> Vec<SpikeEvent> {
        // 1. Reintegrar eventos del overflow
        if !self.overflow.is_empty() {
            let horizon = self.current_tick + self.wheel_size as u64;
            let mut i = 0;
            while i < self.overflow.len() {
                if self.overflow[i].timestamp < horizon {
                    let ev = self.overflow.swap_remove(i);
                    let slot_idx = (ev.timestamp % self.wheel_size as u64) as usize;
                    let phase_idx = (ev.phase_offset & 0x0F) as usize;
                    self.slots[slot_idx][phase_idx].push(ev);
                } else {
                    i += 1;
                }
            }
        }

        // 2. Extraer los eventos de los 16 sub-slots del tick actual
        let slot_idx = (self.current_tick % self.wheel_size as u64) as usize;
        
        // Optimización: Si no hay eventos en ningún sub-slot, evitar asignaciones
        let mut has_events = false;
        for phase_idx in 0..16 {
            if !self.slots[slot_idx][phase_idx].is_empty() {
                has_events = true;
                break;
            }
        }

        if !has_events {
            return Vec::new();
        }

        let mut active_events = Vec::with_capacity(4); // Pequeña capacidad inicial
        for phase_idx in 0..16 {
            if !self.slots[slot_idx][phase_idx].is_empty() {
                let mut events = std::mem::take(&mut self.slots[slot_idx][phase_idx]);
                active_events.append(&mut events);
            }
        }
        active_events
    }

    /// Avanza el reloj de simulación.
    pub fn advance_tick(&mut self, delta: u64) {
        self.current_tick += delta;
    }

    pub fn is_empty(&self) -> bool {
        self.overflow.is_empty() && self.slots.iter().all(|sub_slots| sub_slots.iter().all(|v| v.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_wheel_basic() {
        let mut wheel = TimingWheel::new(10);
        
        // Evento en el futuro cercano (Tick 2)
        wheel.push(SpikeEvent { timestamp: 2, phase_offset: 0, intensity: 1.0, source_neuron_id: 1, target_layer_id: 0, target_neuron_id: 1 });
        
        // Evento en el futuro lejano (Tick 12) -> Overflow inicial
        wheel.push(SpikeEvent { timestamp: 12, phase_offset: 0, intensity: 1.0, source_neuron_id: 2, target_layer_id: 0, target_neuron_id: 2 });

        // Eventos con Phase Coding en el mismo Tick (Tick 5)
        wheel.push(SpikeEvent { timestamp: 5, phase_offset: 10, intensity: 1.0, source_neuron_id: 3, target_layer_id: 0, target_neuron_id: 3 });
        wheel.push(SpikeEvent { timestamp: 5, phase_offset: 2, intensity: 1.0, source_neuron_id: 4, target_layer_id: 0, target_neuron_id: 4 });

        // Tick 0
        assert!(wheel.pop_active().is_empty());
        
        // Tick 2
        wheel.advance_tick(2);
        let events = wheel.pop_active();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timestamp, 2);

        // Tick 5: Deberían salir en orden de fase (2 antes que 10)
        wheel.advance_tick(3);
        let events_tick5 = wheel.pop_active();
        assert_eq!(events_tick5.len(), 2);
        assert_eq!(events_tick5[0].phase_offset, 2);
        assert_eq!(events_tick5[1].phase_offset, 10);

        // Tick 12 (El evento de overflow debería haberse movido a la rueda)
        wheel.advance_tick(7);
        let events = wheel.pop_active();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timestamp, 12);
    }
}
