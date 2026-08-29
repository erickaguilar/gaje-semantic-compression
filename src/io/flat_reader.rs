use crate::core::gtok::GtokNativeTokenizer;
use crate::io::config::ModelConfig;
#[cfg(feature = "native")]
use crate::io::db_loader::NativeLoader;
use crate::io::header::FlatHeaderV2;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FlatTensorEntry {
    pub name: String,
    pub bit_depth: usize,
    pub out_features: usize,
    pub in_features: usize,
    pub dna_off: usize,
    pub dna_len: usize,
    pub c_off: usize,
    pub c_len: usize,
    pub anc_off: usize,
    pub anc_len: usize,
    pub bias_off: usize,
    pub bias_len: usize,
}

#[derive(Clone)]
pub enum FlatBufferSource {
    #[cfg(feature = "native")]
    Mmap(Arc<memmap2::Mmap>),
    Bytes(Arc<[u8]>),
}

impl FlatBufferSource {
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(feature = "native")]
            FlatBufferSource::Mmap(m) => &m[..],
            FlatBufferSource::Bytes(b) => &b[..],
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

pub struct GajeFlatFileReader {
    pub source: FlatBufferSource,
    pub weights_offset: usize,
    pub tensor_map: std::collections::HashMap<String, FlatTensorEntry>,
    pub metadata_json: String,
    pub header: FlatHeaderV2,
}

impl GajeFlatFileReader {
    #[cfg(feature = "native")]
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        Self::spawn_warmup(&mmap);
        Self::from_source(FlatBufferSource::Mmap(mmap))
    }

    /// Precarga en background las páginas del mmap para que la primera inferencia
    /// no pague los page-faults aleatorios (observado: ~22 s en un modelo de
    /// 2.4 GB tras reiniciar el servidor). Combina MADV_WILLNEED (readahead del
    /// kernel) con un toque explícito por página (fault-in garantizado).
    ///
    /// Se puede desactivar con GAJE_MMAP_WARMUP=0.
    #[cfg(feature = "native")]
    fn spawn_warmup(mmap: &std::sync::Arc<memmap2::Mmap>) {
        use std::sync::atomic::{AtomicU64, Ordering};

        if std::env::var("GAJE_MMAP_WARMUP").as_deref() == Ok("0") {
            return;
        }

        // Hint al kernel: readahead del mapeo completo (no bloquea).
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::WillNeed);

        let mmap = std::sync::Arc::clone(mmap);
        let spawned = std::thread::Builder::new()
            .name("gaje-mmap-warmup".into())
            .spawn(move || {
                let start = std::time::Instant::now();
                let slice = &mmap[..];
                let len = slice.len();
                const PAGE: usize = 4096;
                // Contador atómico como black-box: impide que el optimizador
                // elimine las lecturas de las páginas.
                let touched = AtomicU64::new(0);
                let mut offset = 0usize;
                while offset < len {
                    touched.fetch_add(slice[offset] as u64, Ordering::Relaxed);
                    offset += PAGE;
                }
                let pages = (len / PAGE) + 1;
                let checksum = touched.load(Ordering::Relaxed);
                println!(
                    "🔥 [Warm-up mmap] {} páginas ({:.1} GB) precargadas en {:.2}s (checksum interno: {}) — primera inferencia sin penalización de page-faults",
                    pages,
                    len as f32 / 1024.0 / 1024.0 / 1024.0,
                    start.elapsed().as_secs_f32(),
                    checksum % 1000
                );
            });
        if spawned.is_err() {
            eprintln!(
                "⚠️ [Warm-up mmap] No se pudo crear el hilo de precarga; continuando sin warm-up"
            );
        }
    }

    #[cfg(not(feature = "native"))]
    pub fn open(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> std::io::Result<Self> {
        Self::from_source(FlatBufferSource::Bytes(Arc::from(bytes.into_boxed_slice())))
    }

    pub fn from_arc_bytes(bytes: Arc<[u8]>) -> std::io::Result<Self> {
        Self::from_source(FlatBufferSource::Bytes(bytes))
    }

    pub fn from_source(source: FlatBufferSource) -> std::io::Result<Self> {
        let slice = source.as_slice();
        if slice.len() < 4096 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "El buffer es menor a la cabecera de 4096 bytes",
            ));
        }

        let header = FlatHeaderV2::from_bytes(&slice[..4096])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let num_tensors = header.num_tensors as usize;
        let meta_len = header.meta_len as usize;
        let dir_len = header.dir_len as usize;
        let weights_offset = header.weights_offset as usize;

        let meta_start = 4096;
        let meta_end = meta_start + meta_len;
        if meta_end > slice.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Metadata excede el tamaño del buffer",
            ));
        }
        let metadata_json = std::str::from_utf8(&slice[meta_start..meta_end])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            .to_string();

        let dir_start = meta_end;
        let dir_end = dir_start + dir_len;
        if dir_end > slice.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Directorio de tensores excede el tamaño del buffer",
            ));
        }
        let dir_entries: Vec<FlatTensorEntry> = serde_json::from_slice(&slice[dir_start..dir_end])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut tensor_map = std::collections::HashMap::with_capacity(num_tensors);
        for entry in dir_entries {
            tensor_map.insert(entry.name.clone(), entry);
        }

        Ok(Self {
            source,
            weights_offset,
            tensor_map,
            metadata_json,
            header,
        })
    }

    /// Retorna si el archivo .flat contiene un tokenizador GTOK incrustado
    pub fn has_embedded_tokenizer(&self) -> bool {
        self.header.gtok_len > 0
    }

    /// Obtiene la referencia directa a los bytes del tokenizador GTOK incrustado
    pub fn get_embedded_gtok_bytes(&self) -> Option<&[u8]> {
        if self.header.gtok_len == 0 {
            return None;
        }
        let start = self.header.gtok_offset as usize;
        let end = start + self.header.gtok_len as usize;
        let slice = self.source.as_slice();
        if end <= slice.len() {
            Some(&slice[start..end])
        } else {
            None
        }
    }

    /// Deserializa el tokenizador GTOK nativo directamente desde el mapeo de memoria
    pub fn get_embedded_gtok(&self) -> Option<GtokNativeTokenizer> {
        self.get_embedded_gtok_bytes()
            .and_then(|bytes| GtokNativeTokenizer::from_bytes(bytes).ok())
    }

    pub fn get_slice(&self, off: usize, len: usize) -> &[u8] {
        if len == 0 {
            return &[];
        }
        let start = self.weights_offset + off;
        &self.source.as_slice()[start..start + len]
    }

    pub fn get_f32_slice(&self, off: usize, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }
        let bytes = self.get_slice(off, len);
        let count = bytes.len() / 4;
        let mut res = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), res.as_mut_ptr() as *mut u8, bytes.len());
        }
        res
    }

    pub fn get_linear(&self, name: &str, block_size: usize) -> std::io::Result<GenomicLinear> {
        let entry = self.tensor_map.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Flat tensor entry '{}' not found", name),
            )
        })?;

        let dna = self.get_slice(entry.dna_off, entry.dna_len).to_vec();
        let centroids = self.get_f32_slice(entry.c_off, entry.c_len);
        let anchors = self.get_slice(entry.anc_off, entry.anc_len).to_vec();
        let bias = self.get_f32_slice(entry.bias_off, entry.bias_len);

        Ok(GenomicLinear::new(
            dna,
            anchors,
            centroids,
            entry.out_features,
            entry.in_features,
            block_size,
            Vec::new(),
            1e-6,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            bias,
            entry.bit_depth as u8,
        ))
    }

    pub fn load_config(&self) -> std::io::Result<ModelConfig> {
        serde_json::from_str(&self.metadata_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    pub fn load_genomic(&self) -> std::io::Result<GenomicLLM> {
        let mut config = self.load_config()?;

        // Override using ArchitectureDescriptor if present in binary header
        if let Some(desc) = self.header.architecture_descriptor() {
            println!("🧬 [ArchitectureDescriptor] Detectada arquitectura {:?} desde la cabecera binaria (.flat)", desc.family);
            config.n_embd = desc.n_embd;
            config.n_head = desc.n_head;
            config.n_head_kv = desc.n_head_kv;
            config.n_blocks = desc.n_blocks;
            config.config.rope_base = desc.rope_base;
            config.config.rope_style = desc.rope_style;
            config.config.ffn_act = desc.ffn_act;
            config.config.unpermute_weights = desc.qk_permute;
        }

        let block_size = 32;
        let head_dim = config.n_embd / config.n_head;

        let embd_dna = self.get_linear("token_embd", block_size)?;
        let output_norm_entry = self.tensor_map.get("output_norm").ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Tensor 'output_norm' no encontrado en el archivo .flat",
            )
        })?;
        let output_norm = self.get_f32_slice(output_norm_entry.dna_off, output_norm_entry.dna_len);
        let lm_head = self.get_linear("lm_head", block_size)?;

        let mut blocks = Vec::with_capacity(config.n_blocks);
        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);

            let attn_norm_entry =
                self.tensor_map
                    .get(&format!("{}attn_norm", p))
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("Tensor '{}attn_norm' no encontrado", p),
                        )
                    })?;
            let attn_norm = self.get_f32_slice(attn_norm_entry.dna_off, attn_norm_entry.dna_len);

            let ffn_norm_entry =
                self.tensor_map
                    .get(&format!("{}ffn_norm", p))
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("Tensor '{}ffn_norm' no encontrado", p),
                        )
                    })?;
            let ffn_norm = self.get_f32_slice(ffn_norm_entry.dna_off, ffn_norm_entry.dna_len);

            let has_fused_qkv = self.tensor_map.contains_key(&format!("{}attn_qkv", p));
            let has_fused_gate_up = self.tensor_map.contains_key(&format!("{}ffn_gate_up", p));

            let (q_gen, k_gen, v_gen, fused_qkv) = if has_fused_qkv {
                let f_qkv = self.get_linear(&format!("{}attn_qkv", p), block_size)?;
                (
                    GenomicLinear::empty(),
                    GenomicLinear::empty(),
                    GenomicLinear::empty(),
                    Some(f_qkv),
                )
            } else {
                (
                    self.get_linear(&format!("{}attn_q", p), block_size)?,
                    self.get_linear(&format!("{}attn_k", p), block_size)?,
                    self.get_linear(&format!("{}attn_v", p), block_size)?,
                    None,
                )
            };

            let w_o = self.get_linear(&format!("{}attn_output", p), block_size)?;

            let (gate_gen, up_gen, fused_gate_up) = if has_fused_gate_up {
                let f_gu = self.get_linear(&format!("{}ffn_gate_up", p), block_size)?;
                (GenomicLinear::empty(), GenomicLinear::empty(), Some(f_gu))
            } else {
                (
                    self.get_linear(&format!("{}ffn_gate", p), block_size)?,
                    self.get_linear(&format!("{}ffn_up", p), block_size)?,
                    None,
                )
            };

            let w_down = self.get_linear(&format!("{}ffn_down", p), block_size)?;

            let attn = GenomicAttention::new(
                config.n_head,
                config.n_head_kv,
                head_dim,
                attn_norm,
                config.eps,
                config.config.rope_base,
                config.config.rope_style.clone(),
            );

            let mut block = RustGenomicBlock::new(
                i,
                attn,
                q_gen,
                k_gen,
                v_gen,
                w_o,
                gate_gen,
                up_gen,
                w_down,
                ffn_norm,
                config.eps,
                config.config.ffn_act.clone(),
                config.config.use_genomic_norm,
                1.0,
                config.config.rna_threshold,
            );
            block.fused_qkv = fused_qkv;
            block.fused_gate_up = fused_gate_up;
            blocks.push(block);
        }

        Ok(GenomicLLM {
            embeddings: embd_dna,
            blocks,
            output_norm,
            lm_head,
            eps: config.eps,
            k_wta_ratio: 0.50,
            topology: None,
            quantum_embeddings: None,
            gpu_layers: 0,
            use_gpu: false,
        })
    }
}

pub fn load_genomic_auto(path: &str) -> std::io::Result<GenomicLLM> {
    if let Ok(mut f) = std::fs::File::open(path) {
        use std::io::Read;
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok() && &magic == b"GAJE" {
            println!("⚡ [Zero-Copy Mmap] Detectado formato binario plano .gaje.flat. Carga mmap instantánea...");
            let reader = GajeFlatFileReader::open(path)?;
            return reader.load_genomic();
        }
    }
    #[cfg(feature = "native")]
    {
        let loader = NativeLoader::new(path)?;
        loader.load_llm()
    }
    #[cfg(not(feature = "native"))]
    {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Formato no .flat no soportado en modo sin base nativa",
        ))
    }
}
