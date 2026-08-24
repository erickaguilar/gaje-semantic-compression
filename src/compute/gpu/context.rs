// =============================================================================
// context.rs — Contexto e Inicialización de GPU WGPU/Vulkan en GAJE Helix
// =============================================================================

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuDeviceInfo {
    pub device_name: String,
    pub backend: String,
    pub device_type: String,
    pub is_unified_memory: bool,
    pub max_buffer_size_mb: f64,
    pub max_compute_workgroups_per_dim: [u32; 3],
}

#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub info: GpuDeviceInfo,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Inicializa el contexto de GPU buscando adaptadores Vulkan / Primarios compatibles.
    pub fn init() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "No compatible GPU adapter found (Vulkan/DirectX/Metal)".to_string())?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let is_uma = match adapter_info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => true,
            _ => false,
        };

        let backend_str = format!("{:?}", adapter_info.backend);
        let device_type_str = format!("{:?}", adapter_info.device_type);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("GAJE Helix GPU Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("Failed to request GPU device: {:?}", e))?;

        let info = GpuDeviceInfo {
            device_name: adapter_info.name.clone(),
            backend: backend_str,
            device_type: device_type_str,
            is_unified_memory: is_uma,
            max_buffer_size_mb: limits.max_buffer_size as f64 / (1024.0 * 1024.0),
            max_compute_workgroups_per_dim: [
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ],
        };

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
        })
    }
}

// Global Singleton GpuContext
#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    pub static ref GLOBAL_GPU_CONTEXT: Option<Arc<GpuContext>> = {
        match GpuContext::init() {
            Ok(ctx) => {
                eprintln!("🎮 [GPU Compute] Inicializado adaptador: {} ({})", ctx.info.device_name, ctx.info.backend);
                Some(Arc::new(ctx))
            }
            Err(e) => {
                eprintln!("⚠️ [GPU Compute] No se pudo inicializar GPU: {}. Usando fallback CPU SIMD.", e);
                None
            }
        }
    };
}

/// Consulta si el dispositivo tiene GPU activa y lista para cómputo.
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GLOBAL_GPU_CONTEXT.is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Obtiene la información del dispositivo GPU detectado.
pub fn get_gpu_info() -> Option<GpuDeviceInfo> {
    #[cfg(feature = "gpu")]
    {
        GLOBAL_GPU_CONTEXT.as_ref().map(|ctx| ctx.info.clone())
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}
