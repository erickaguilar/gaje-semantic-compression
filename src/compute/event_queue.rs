use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Representa un evento de disparo (Spike) en el tiempo con precisión de fase e intensidad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeEvent {
    pub timestamp: u64,           // Tiempo en el que ocurre el evento (ticks de simulación)
    pub phase_offset: u8,         // Desplazamiento de fase (0-15 sub-ticks para precisión temporal)
    pub intensity: f32,           // Intensidad del disparo (Graded Spiking)
    pub source_neuron_id: usize,  // ID de la neurona que disparó
    pub target_layer_id: usize,   // ID de la capa destino
    pub target_neuron_id: usize,  // ID de la neurona destino dentro de la capa
}

/// Implementación de ordenamiento para BinaryHeap (Min-Heap basado en timestamp y fase).
/// Queremos que el evento con el menor tiempo y fase sea el primero en salir.
impl Eq for SpikeEvent {}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Invertimos el orden para que sea un Min-Heap
        // Primero comparamos el timestamp
        let res = other.timestamp.cmp(&self.timestamp);
        if res == Ordering::Equal {
            // Si el tiempo es igual, comparamos la fase (latencia)
            other.phase_offset.cmp(&self.phase_offset)
        } else {
            res
        }
    }
}

/// Cola de prioridad para la gestión de eventos neuromórficos.
pub struct EventQueue {
    queue: BinaryHeap<SpikeEvent>,
    pub current_tick: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            current_tick: 0,
        }
    }

    /// Encola un nuevo evento.
    pub fn push(&mut self, event: SpikeEvent) {
        self.queue.push(event);
    }

    /// Obtiene el siguiente evento si su timestamp es <= current_tick.
    pub fn pop_active(&mut self) -> Option<SpikeEvent> {
        if let Some(event) = self.queue.peek() {
            if event.timestamp <= self.current_tick {
                return self.queue.pop();
            }
        }
        None
    }

    /// Avanza el reloj de simulación al siguiente evento o por un delta fijo.
    pub fn advance_tick(&mut self, delta: u64) {
        self.current_tick += delta;
    }

    /// Salta directamente al tiempo del próximo evento (aceleración de simulación).
    pub fn skip_to_next_event(&mut self) {
        if let Some(event) = self.queue.peek() {
            if event.timestamp > self.current_tick {
                self.current_tick = event.timestamp;
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
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
        eq.push(SpikeEvent { timestamp: 15, phase_offset: 0, intensity: 1.0, source_neuron_id: 3, target_layer_id: 0, target_neuron_id: 3 });

        eq.advance_tick(5);
        // Debería salir primero el de phase 2 (menor latencia)
        let ev1 = eq.pop_active().unwrap();
        assert_eq!(ev1.timestamp, 5);
        assert_eq!(ev1.phase_offset, 2);
        
        let ev2 = eq.pop_active().unwrap();
        assert_eq!(ev2.timestamp, 5);
        assert_eq!(ev2.phase_offset, 5);

        assert!(eq.pop_active().is_none());

        eq.advance_tick(5); // Tiempo 10
        let ev3 = eq.pop_active().unwrap();
        assert_eq!(ev3.timestamp, 10);

        eq.skip_to_next_event(); // Salta a 15
        assert_eq!(eq.current_tick, 15);
        let ev4 = eq.pop_active().unwrap();
        assert_eq!(ev4.timestamp, 15);
    }
}
