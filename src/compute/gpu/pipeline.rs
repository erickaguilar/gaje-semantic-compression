// =============================================================================
// pipeline.rs — Pipeline y Despachador de Cómputo WGSL para GPU
// =============================================================================

#[cfg(feature = "gpu")]
use crate::compute::gpu::context::{GpuContext, GLOBAL_GPU_CONTEXT};
#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
pub struct GpuComputePipelines {
    pub ctx: Arc<GpuContext>,
    pub gemv_f32_pipeline: wgpu::ComputePipeline,
    pub swiglu_pipeline: wgpu::ComputePipeline,
    pub rms_norm_pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
impl GpuComputePipelines {
    pub fn new(ctx: Arc<GpuContext>) -> Result<Self, String> {
        let device = &ctx.device;

        // Compile Shaders
        let gemv_f32_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMV FP32 WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_f32.wgsl").into()),
        });

        let swiglu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SwiGLU WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/swiglu.wgsl").into()),
        });

        let rms_norm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMS Norm WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rms_norm.wgsl").into()),
        });

        // Create Pipelines
        let gemv_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMV FP32 Pipeline"),
            layout: None,
            module: &gemv_f32_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SwiGLU Pipeline"),
            layout: None,
            module: &swiglu_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rms_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMS Norm Pipeline"),
            layout: None,
            module: &rms_norm_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            ctx,
            gemv_f32_pipeline,
            swiglu_pipeline,
            rms_norm_pipeline,
        })
    }

    /// Ejecuta SwiGLU en GPU y retorna el vector resultante en memoria CPU.
    pub fn execute_swiglu(
        &self,
        gate: &[f32],
        up: &[f32],
        h_scale: f32,
    ) -> Result<Vec<f32>, String> {
        let len = gate.len();
        if len != up.len() {
            return Err("Gate and Up length mismatch".to_string());
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SwigluUniforms {
            len: u32,
            h_scale: f32,
            _pad: [u32; 2],
        }

        let uniforms = SwigluUniforms {
            len: len as u32,
            h_scale,
            _pad: [0, 0],
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SwiGLU Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let gate_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gate Buffer"),
            contents: bytemuck::cast_slice(gate),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let up_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Up Buffer"),
            contents: bytemuck::cast_slice(up),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_byte_size = (len * std::mem::size_of::<f32>()) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.swiglu_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SwiGLU Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: up_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SwiGLU Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SwiGLU Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.swiglu_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (len as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_byte_size);
        queue.submit(Some(encoder.finish()));

        // Readback synchronous via pollster
        let slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive map event: {:?}", e))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        Ok(result)
    }

    /// Multiplicación Matriz-Vector FP32 en GPU: y = W * x
    pub fn execute_gemv_f32(
        &self,
        weights: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        if weights.len() != rows * cols {
            return Err("Weights matrix dimensions mismatch".to_string());
        }
        if x.len() != cols {
            return Err("Input vector length mismatch".to_string());
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GemvUniforms {
            rows: u32,
            cols: u32,
            _pad: [u32; 2],
        }

        let uniforms = GemvUniforms {
            rows: rows as u32,
            cols: cols as u32,
            _pad: [0, 0],
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GEMV Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GEMV Weights Buffer"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GEMV X Buffer"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_byte_size = (rows * std::mem::size_of::<f32>()) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GEMV Output Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GEMV Readback Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.gemv_f32_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GEMV Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GEMV Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMV Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.gemv_f32_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (rows as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_byte_size);
        queue.submit(Some(encoder.finish()));

        // Readback
        let slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive map event: {:?}", e))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        Ok(result)
    }

    /// Normalización RMS en GPU
    pub fn execute_rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let len = x.len();
        if len != weight.len() {
            return Err("Input and Weight vector length mismatch in RMSNorm".to_string());
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RmsUniforms {
            len: u32,
            eps: f32,
            _pad: [u32; 2],
        }

        let uniforms = RmsUniforms {
            len: len as u32,
            eps,
            _pad: [0, 0],
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm X Buffer"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Weight Buffer"),
            contents: bytemuck::cast_slice(weight),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_byte_size = (len * std::mem::size_of::<f32>()) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RMSNorm Output Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RMSNorm Readback Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.rms_norm_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RMSNorm Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RMSNorm Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RMSNorm Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.rms_norm_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_byte_size);
        queue.submit(Some(encoder.finish()));

        // Readback
        let slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive map event: {:?}", e))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        Ok(result)
    }
}

// Global Singleton Pipelines
#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    pub static ref GLOBAL_GPU_PIPELINES: Option<Arc<GpuComputePipelines>> = {
        GLOBAL_GPU_CONTEXT.as_ref().and_then(|ctx| {
            match GpuComputePipelines::new(ctx.clone()) {
                Ok(pipes) => {
                    eprintln!("🚀 [GPU Pipelines] Shaders de Cómputo WGSL compilados y listos en GPU.");
                    Some(Arc::new(pipes))
                }
                Err(e) => {
                    eprintln!("⚠️ [GPU Pipelines] Error compilando pipelines WGSL: {}. Fallback a CPU.", e);
                    None
                }
            }
        })
    };
}

/// Helper para ejecutar SwiGLU en GPU si está disponible
pub fn gpu_swiglu(gate: &[f32], up: &[f32], h_scale: f32) -> Option<Vec<f32>> {
    #[cfg(feature = "gpu")]
    {
        GLOBAL_GPU_PIPELINES
            .as_ref()
            .and_then(|p| p.execute_swiglu(gate, up, h_scale).ok())
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// Helper para ejecutar GEMV FP32 en GPU si está disponible
pub fn gpu_gemv_f32(weights: &[f32], x: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    #[cfg(feature = "gpu")]
    {
        GLOBAL_GPU_PIPELINES
            .as_ref()
            .and_then(|p| p.execute_gemv_f32(weights, x, rows, cols).ok())
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// Helper para ejecutar RMSNorm en GPU si está disponible
pub fn gpu_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Option<Vec<f32>> {
    #[cfg(feature = "gpu")]
    {
        GLOBAL_GPU_PIPELINES
            .as_ref()
            .and_then(|p| p.execute_rms_norm(x, weight, eps).ok())
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}
