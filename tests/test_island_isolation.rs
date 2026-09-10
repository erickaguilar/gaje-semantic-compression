//! Test de Integración: Aislamiento de Ruido y Fidelidad de Memoria en el Island Model (.gmem)
//! Valida los 4 mecanismos matemáticos de mitigación descritos en:
//! docs/research/ISLAND_MEMORY_FIDELITY_AND_NOISE_ISOLATION_FINDINGS.md

use _impl::compute::island::{IslandNiche, IslandOrchestrator, IslandSearchResult};

#[test]
fn test_entropy_gap_gating_protects_against_diffuse_noise() {
    let dim = 16;
    let mut orch = IslandOrchestrator::new(dim);

    // Vector base ortonormal
    let mut v1 = vec![0.0f32; dim as usize];
    v1[0] = 1.0;
    let mut v2 = vec![0.0f32; dim as usize];
    v2[1] = 1.0;

    orch.add_memory(
        IslandNiche::Conversational,
        1,
        v1.clone(),
        "Conversación casual sobre el clima ayer.".to_string(),
    );
    orch.add_memory(
        IslandNiche::Conversational,
        2,
        v2.clone(),
        "Comentario acerca del tráfico por la mañana.".to_string(),
    );

    // Query equidistante y difusa: v_q = (0.707, 0.707, 0, ...)
    // Ambos recuerdos tendrán similitud casi idéntica ~0.707 * 0.8 = 0.565 -> Delta_top ~ 0 < 0.12
    let mut q_diffuse = vec![0.0f32; dim as usize];
    q_diffuse[0] = 0.7071;
    q_diffuse[1] = 0.7071;

    let prompt = "¿Cuál es la velocidad de la luz?";
    let augmented = orch.build_augmented_prompt(prompt, &q_diffuse, 100);

    // El orquestador debe abortar la inyección por ruido difuso
    assert_eq!(
        augmented, prompt,
        "La inyección debió ser abortada ante ruido difuso con baja brecha de entropía"
    );
}

#[test]
fn test_factual_retrieval_with_clear_entropy_gap() {
    let dim = 8;
    let mut orch = IslandOrchestrator::new(dim);

    let mut doc_v = vec![0.0f32; dim as usize];
    doc_v[0] = 1.0;
    orch.add_memory(
        IslandNiche::Documental,
        42,
        doc_v.clone(),
        "La constante de Planck reducida es hbar = 1.054571817e-34 J*s.".to_string(),
    );

    let mut other_v = vec![0.0f32; dim as usize];
    other_v[3] = 1.0;
    orch.add_memory(
        IslandNiche::Conversational,
        99,
        other_v,
        "El café estuvo delicioso.".to_string(),
    );

    // Query altamente alineada con el hecho documental
    let mut q_sharp = vec![0.0f32; dim as usize];
    q_sharp[0] = 0.98;
    q_sharp[1] = 0.10;

    let prompt = "¿Cuál es el valor de hbar?";
    let augmented = orch.build_augmented_prompt(prompt, &q_sharp, 100);

    assert!(
        augmented.contains("Contexto de Memoria Recolectado:"),
        "Debe inyectar contexto cuando hay resonancia fáctica clara"
    );
    assert!(
        augmented.contains("constante de Planck"),
        "Debe contener la memoria fáctica documental"
    );
    assert!(
        !augmented.contains("El café estuvo delicioso"),
        "El recuerdo irrelevante debe estar completamente excluido"
    );
}

#[test]
fn test_kwta_lateral_inhibition_prunes_subdominant_memories() {
    let dim = 8;
    let orch = IslandOrchestrator::new(dim);

    // Simulamos 3 matches recuperados
    let matches = vec![
        IslandSearchResult {
            niche: IslandNiche::Documental,
            id: 101,
            similarity: 0.96, // Pico dominante
            text: "GAJE implementa compresión genética con kernels SIMD.".to_string(),
        },
        IslandSearchResult {
            niche: IslandNiche::Documental,
            id: 102,
            similarity: 0.90, // 0.90 >= 0.90 * 0.96 (0.864) -> Retenido por K-WTA
            text: "El motor Helix opera con latencia sub-milisegundo.".to_string(),
        },
        IslandSearchResult {
            niche: IslandNiche::Documental,
            id: 103,
            similarity: 0.84, // 0.84 < 0.864 -> Podado por K-WTA a pesar de superar 0.82
            text: "Documento genérico sin resonancia pico.".to_string(),
        },
    ];

    let prompt = "¿Cómo opera el núcleo nativo de GAJE?";
    let augmented = orch.build_augmented_prompt_from_matches(prompt, &matches, 200);

    assert!(augmented.contains("GAJE implementa compresión genética"));
    assert!(augmented.contains("El motor Helix opera con latencia sub-milisegundo"));
    assert!(
        !augmented.contains("Documento genérico sin resonancia pico"),
        "K-WTA debe podar el match que cae bajo el 90% de la resonancia máxima"
    );
}

#[test]
fn test_niche_decoupling_with_orthogonal_projection() {
    let dim = 8;
    let mut orch = IslandOrchestrator::new(dim);

    // Vector canónico v = [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]
    let mut raw_v = vec![0.0f32; dim as usize];
    for i in 0..4 {
        raw_v[i] = 0.5;
    }

    // Registramos en documental y conversacional usando proyección ortogonal
    orch.add_memory_orthogonal(
        IslandNiche::Documental,
        1,
        &raw_v,
        "Fórmula fáctica de cuantización cuaternaria.".to_string(),
    );
    orch.add_memory_orthogonal(
        IslandNiche::Conversational,
        2,
        &raw_v,
        "Diálogo casual usando las mismas palabras clave.".to_string(),
    );

    // Consulta con proyección ortogonal
    let res = orch.retrieve_context_orthogonal(&raw_v, 2);

    // Ambas islas deben recuperar sus respectivos recuerdos desacoplados
    let has_doc = res.iter().any(|r| r.niche == IslandNiche::Documental && r.id == 1);
    let has_conv = res.iter().any(|r| r.niche == IslandNiche::Conversational && r.id == 2);

    assert!(has_doc, "Debe recuperar el recuerdo documental proyectado");
    assert!(has_conv, "Debe recuperar el recuerdo conversacional proyectado");
}
