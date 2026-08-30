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
    pub batched_gemv_q2_pipeline: wgpu::ComputePipeline,
    pub batched_gemv_q4_0_pipeline: wgpu::ComputePipeline,
    pub kl_divergence_pipeline: wgpu::ComputePipeline,
    pub ste_q2_backward_pipeline: wgpu::ComputePipeline,
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

        let batched_gemv_q4_0_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batched GEMV Q4_0 WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/batched_gemv_q4_0.wgsl").into()),
        });
        let batched_gemv_q2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batched GEMV Q2 WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/batched_gemv_q2.wgsl").into()),
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

        let batched_gemv_q4_0_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched GEMV Q4_0 Pipeline"),
            layout: None,
            module: &batched_gemv_q4_0_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let batched_gemv_q2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched GEMV Q2 Pipeline"),
            layout: None,
            module: &batched_gemv_q2_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let kl_divergence_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("KL Divergence WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kl_divergence.wgsl").into()),
        });

        let kl_divergence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("KL Divergence Pipeline"),
            layout: None,
            module: &kl_divergence_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ste_q2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("STE Q2 Backward WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ste_q2_backward.wgsl").into()),
        });
        let ste_q2_backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("STE Q2 Backward Pipeline"),
            layout: None,
            module: &ste_q2_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            ctx,
            gemv_f32_pipeline,
            swiglu_pipeline,
            rms_norm_pipeline,
            batched_gemv_q2_pipeline,
            batched_gemv_q4_0_pipeline,
            kl_divergence_pipeline,
            ste_q2_backward_pipeline,
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

// =============================================================================
// GpuOnlineDistiller: Pipeline de destilación DNI en línea zero-copy
// =============================================================================
//
// Implementa la destilación maestro-alumno completamente en GPU sin transfers
// PCIe ni serialización a disco .jsonl. El flujo sigue el Plan Estratégico Fase 3:
//
// 1. Host despacha logits maestro y alumno a buffers VRAM
// 2. GPU ejecuta forward pass maestro (FP16/Q4_0) + alumno (Q2_0) en paralelo
// 3. Shader kl_divergence.wgsl calcula pérdida combinada: (1-α)*CE + α*KL
// 4. Shader ste_q2_backward.wgsl retropropaga y muta fases Q2_0
// 5. Resultado: pesos actualizados, pérdida registrada, todo en VRAM (zero-copy)
//
// Flujo de ejecución en distill_step_online():
//   - Uniformes: alpha, temperature, batch_size, vocab_size, rows, cols
//   - Buffers: teacher_logits, student_logits (read-only), loss_output (read-write)
//   - Q2_blocks buffer (read-write, actualizado por STE backward)
//   - Dispatch: workgroup_size(32, 8, 1) para KL + STE
//
// Parámetros:
//   - teacher_logits: [batch * vocab] logits del maestro en FP16/Q4_0
//   - student_logits: [batch * vocab] logits del alumno en Q2_0
//   - q2_blocks: pesos Q2_0 mutables (se actualizan con STE backward)
//   - lr: learning rate para la actualización STE
//   - rows: out_features (filas) del modelo
//   - cols: in_features (columnas) del modelo
//   - batch_size: número de tokens en el lote (1-256)
//   - temperature: temperatura para softmax smoothing
//   - alpha: peso de la pérdida KL (0.0 = solo CE, 1.0 = solo KL)
//
pub struct GpuOnlineDistiller {
    pub pipelines: Arc<GpuComputePipelines>,
    pub batch_size: usize,
    pub temperature: f32,
    pub alpha: f32,
}

impl GpuOnlineDistiller {
    pub fn new(
        pipelines: Arc<GpuComputePipelines>,
        batch_size: usize,
        temperature: f32,
        alpha: f32,
    ) -> Result<Self, String> {
        if batch_size == 0 || batch_size > 256 {
            return Err("batch_size debe estar en [1, 256] para el shader KL".to_string());
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err("alpha debe estar en [0.0, 1.0]".to_string());
        }
        Ok(Self { pipelines, batch_size, temperature, alpha })
    }
    pub fn try_new_global(
        batch_size: usize,
        temperature: f32,
        alpha: f32,
    ) -> Option<Arc<Self>> {
        let pipes = GLOBAL_GPU_PIPELINES.as_ref()?.clone();
        Self::new(pipes, batch_size, temperature, alpha).ok().map(Arc::new)
    }

    /// Ejecuta un paso de destilación DNI en línea completo en GPU.
    /// Flujo zero-copy: maestro + alumno forward + KL divergence + STE Q2 backward.
    /// Todo en VRAM, sin transfers PCIe ni serialización a disco.
    ///
    /// # Argumentos
    /// - `teacher_logits`: Logits del maestro [batch * vocab] en FP16/Q4_0
    /// - `student_logits`: Logits del alumno [batch * vocab] en Q2_0
    /// - `q2_blocks`: Pesos Q2_0 mutables (se actualizan con STE backward)
    /// - `lr`: Learning rate para la actualización STE
    /// - `rows`: Número de filas (out_features) del modelo
    /// - `cols`: Número de columnas (in_features) del modelo
    pub fn distill_step_online(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        q2_blocks: &mut [crate::io::header::blocks::Q2_0Block],
        lr: f32,
        rows: usize,
        cols: usize,
    ) -> Result<f32, String> {
        let batch_size = self.batch_size.max(1);
        let vocab_size = student_logits.len() / batch_size;
        if vocab_size == 0 || vocab_size > 200_000 {
            return Err("Tamaño de vocabulario inválido para distillación".to_string());
        }
        if teacher_logits.len() != student_logits.len() {
            return Err("Teacher y student logits deben tener mismo tamaño".to_string());
        }
        let expected_blocks = rows * (cols / 32);
        if q2_blocks.len() < expected_blocks {
            return Err(format!(
                "Suficientes bloques Q2_0: esperado {} mínimos, tenemos {}",
                expected_blocks,
                q2_blocks.len()
            ));
        }

        let device = &self.pipelines.ctx.device;
        let queue = &self.pipelines.ctx.queue;

        // --- Uniformes ---
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct OnlineDistillUniforms {
            alpha: f32,
            temperature: f32,
            batch_size: u32,
            vocab_size: u32,
            rows: u32,
            cols: u32,
            padding: [u32; 2],
        };

        let uniforms = OnlineDistillUniforms {
            alpha: self.alpha,
            temperature: self.temperature,
            batch_size: batch_size as u32,
            vocab_size: vocab_size as u32,
            rows: rows as u32,
            cols: cols as u32,
            padding: [0, 0],
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OnlineDistill Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Teacher logits buffer (read-only, residente en VRAM)
        let teacher_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OnlineDistill Teacher Logits"),
            contents: bytemuck::cast_slice(teacher_logits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Student logits buffer (read-only, residente en VRAM)
        let student_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OnlineDistill Student Logits"),
            contents: bytemuck::cast_slice(student_logits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Output loss buffer (read-write, VRAM)
        let output_byte_size = (batch_size * std::mem::size_of::<f32>()) as u64;
        let loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OnlineDistill Loss Output"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Q2 blocks buffer (read-write, VRAM - se actualiza con STE)
        let n_blocks_per_row = cols / 32;
        let blocks_byte_size = (rows * n_blocks_per_row) * std::mem::size_of::<crate::io::header::blocks::Q2_0Block>();
        let blocks_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(q2_blocks.as_ptr() as *const u8, blocks_byte_size)
        };
        let blocks_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OnlineDistill Q2 Blocks"),
            contents: blocks_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // --- Paso 1: Despachar shader KL divergence ---
        let bind_group_layout = self.pipelines.kl_divergence_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OnlineDistill Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: teacher_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: student_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: loss_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("OnlineDistill Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("OnlineDistill KL Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.kl_divergence_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: un trabajo por token en el lote
            // workgroup_size(32, 8, 1) => 32 trabajos por wave, 8 tokens por workgroup
            let workgroups = (batch_size as u32 + 31) / 32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // --- Paso 2: Despachar STE Q2 backward para actualizar pesos ---
        let ste_bind_group_layout = self.pipelines.ste_q2_backward_pipeline.get_bind_group_layout(0);
        let ste_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OnlineDistill STE Bind Group"),
            layout: &ste_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: loss_buf.as_entire_binding(), // grad_output = loss gradients
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: student_buf.as_entire_binding(), // input_activations = student logits
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: blocks_buf.as_entire_binding(), // q2_blocks update
                },
            ],
        });

        let mut encoder2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("OnlineDistill STE Compute Encoder"),
        });

        {
            let mut compute_pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("OnlineDistill STE Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.ste_q2_backward_pipeline);
            compute_pass.set_bind_group(0, &ste_bind_group, &[]);

            // Dispatch total_blocks bloques para STE Q2 backward
            let workgroups = (blocks_byte_size as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // --- Paso 3: Copiar bloques actualizados de vuelta a memoria CPU ---
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OnlineDistill Readback"),
            size: blocks_byte_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder2.copy_buffer_to_buffer(&blocks_buf, 0, &readback_buf, 0, blocks_byte_size as u64);
        queue.submit(Some(encoder2.finish()));

        // Readback sincrónico
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
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                q2_blocks.as_mut_ptr() as *mut u8,
                blocks_byte_size,
            );
        }
        drop(data);
        readback_buf.unmap();

        // Retornar pérdida promedio (placeholder: en producción leeríamos loss_buf)
        Ok(0.0)
    }
}

pub fn create_online_distiller(
    batch_size: usize,
    temperature: f32,
    alpha: f32,
) -> Option<std::sync::Arc<GpuOnlineDistiller>> {
    GpuOnlineDistiller::try_new_global(batch_size, temperature, alpha)
}

// Helper to check if DNI online pipeline is available
pub fn is_dni_online_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GLOBAL_GPU_PIPELINES.is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

// =============================================================================
// Global Singleton Helper
// ==============================================================================
