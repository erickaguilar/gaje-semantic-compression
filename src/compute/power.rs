use std::collections::BTreeMap;
use std::fs;

/// Representa los tipos de núcleos en una arquitectura ARM big.LITTLE / Cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CpuCluster {
    Little, // Núcleos de eficiencia (Baja frecuencia, bajo consumo)
    Medium, // Núcleos de rendimiento balanceado
    Big,    // Núcleos de máximo rendimiento (Prime cores)
}

/// Gestor de Energía y Afinidad de Núcleos.
pub struct PowerManager {
    clusters: BTreeMap<CpuCluster, Vec<usize>>,
}

impl PowerManager {
    /// Detecta la arquitectura de núcleos leyendo las frecuencias máximas del sistema.
    pub fn detect() -> Self {
        let mut freqs = BTreeMap::new();

        // Leer frecuencias de todos los núcleos disponibles
        for i in 0..32 {
            // Límite razonable de núcleos
            let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", i);
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(freq) = content.trim().parse::<u64>() {
                    freqs.entry(freq).or_insert_with(Vec::new).push(i);
                }
            }
        }

        let mut clusters = BTreeMap::new();
        let unique_freqs: Vec<u64> = freqs.keys().cloned().collect();

        match unique_freqs.len() {
            1 => {
                // Arquitectura uniforme
                clusters.insert(CpuCluster::Big, freqs.values().next().unwrap().clone());
            }
            2 => {
                // big.LITTLE clásico (4+4 o similar)
                clusters.insert(
                    CpuCluster::Little,
                    freqs.get(&unique_freqs[0]).unwrap().clone(),
                );
                clusters.insert(
                    CpuCluster::Big,
                    freqs.get(&unique_freqs[1]).unwrap().clone(),
                );
            }
            _ => {
                // Tri-cluster (Ej: 4+3+1 como en el output previo)
                clusters.insert(
                    CpuCluster::Little,
                    freqs.get(&unique_freqs[0]).unwrap().clone(),
                );
                clusters.insert(
                    CpuCluster::Medium,
                    freqs.get(&unique_freqs[1]).unwrap().clone(),
                );
                clusters.insert(
                    CpuCluster::Big,
                    freqs
                        .get(&unique_freqs[unique_freqs.len() - 1])
                        .unwrap()
                        .clone(),
                );
            }
        }

        Self { clusters }
    }

    /// Asigna el hilo actual a un cluster específico.
    pub fn set_thread_affinity(&self, cluster: CpuCluster) -> Result<(), String> {
        #[cfg(all(target_os = "android", feature = "native"))]
        {
            if let Some(cores) = self.clusters.get(&cluster) {
                unsafe {
                    let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                    libc::CPU_ZERO(&mut cpuset);
                    for &core in cores {
                        libc::CPU_SET(core, &mut cpuset);
                    }

                    let result =
                        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
                    if result == 0 {
                        return Ok(());
                    } else {
                        return Err(format!(
                            "Error en sched_setaffinity: {}",
                            std::io::Error::last_os_error()
                        ));
                    }
                }
            }
            Err("Cluster no disponible".to_string())
        }
        #[cfg(not(all(target_os = "android", feature = "native")))]
        {
            let _ = cluster;
            Ok(()) // No-op en sistemas no compatibles (x86_64, wasm32, etc.)
        }
    }

    /// Retorna un resumen de los clusters detectados.
    pub fn summary(&self) -> String {
        let mut s = String::from("Clusters detectados:\n");
        for (cluster, cores) in &self.clusters {
            s.push_str(&format!("  {:?}: Cores {:?}\n", cluster, cores));
        }
        s
    }
}
