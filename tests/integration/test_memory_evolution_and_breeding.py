#!/usr/bin/env python3
"""
🧬 GAJE HELIX: CERTIFICACIÓN DE FASE 4 — EVOLUCIÓN DE MEMORIA Y BREEDING (.gmem v2)
================================================================================
Valida la hipótesis H3 del plan de épocas de memoria:
1. Creación de dos organismos con especialidades disjuntas (Bio y Comp).
2. Cross-Breeding de memoria entre organismos (Fusión de islas con poda).
3. Evolución genética sobre la capa de memoria DNI (Ajuste óptimo de nichos).
4. Verificación de Gate de Promoción post-cruce con 100% Needle-Recall en ambas disciplinas.
================================================================================
"""

import json
import shutil
import tempfile
import numpy as np

import gaje.core._impl as _impl


def main():
    print(
        "================================================================================"
    )
    print("🧬 GAJE HELIX: CERTIFICACIÓN DE FASE 4 — EVOLUCIÓN Y BREEDING DE MEMORIA")
    print(
        "================================================================================"
    )

    temp_root = tempfile.mkdtemp(prefix="gaje_phase4_breeding_")
    dim = 64

    try:
        # 1. Organismo A: Especialista en Biología Genómica
        mgr_bio = _impl.EpochManager(temp_root, "organism_bio", dim)
        orch_bio = _impl.IslandOrchestrator(dim)

        v_bio = np.random.randn(dim).astype(np.float32)
        v_bio /= np.linalg.norm(v_bio)
        orch_bio.add_memory_py(
            "documental", 101, v_bio.tolist(), "AGUJA_GENOMICA_CRISPR_CAS9"
        )
        ep_bio = mgr_bio.create_snapshot_py(orch_bio, "Génesis Biología", None)
        print(f"[*] 1. Organismo Bio creado: Época {ep_bio} (Recuerdo ID 101)")

        # 2. Organismo B: Especialista en Computación Cuántica / Rust
        mgr_comp = _impl.EpochManager(temp_root, "organism_comp", dim)
        orch_comp = _impl.IslandOrchestrator(dim)

        v_comp = np.random.randn(dim).astype(np.float32)
        v_comp /= np.linalg.norm(v_comp)
        orch_comp.add_memory_py(
            "documental", 202, v_comp.tolist(), "AGUJA_COMPUTACION_RUST_AVX2"
        )
        ep_comp = mgr_comp.create_snapshot_py(orch_comp, "Génesis Computación", None)
        print(f"[*] 2. Organismo Comp creado: Época {ep_comp} (Recuerdo ID 202)")

        # 3. Cross-Breeding: Fusión de recuerdos de Comp hacia Bio
        print("\n[*] 3. Ejecutando Cross-Breeding de Memoria: [Comp] -> [Bio]...")
        stats_json = mgr_bio.merge_memory_islands_py(orch_bio, orch_comp, 0.95)
        stats = json.loads(stats_json)
        print("    - Estadísticas de Fusión:")
        print(json.dumps(stats, indent=4))

        assert stats["total_documental_entries"] == 2
        assert stats["episodic_transferred"] == 1  # 1 entrada transferida

        # 4. Evolución de Pesos de Nicho (Algoritmo Genético DNI)
        print("\n[*] 4. Ejecutando Búsqueda Evolutiva sobre Capa de Memoria DNI...")
        golden_queries = [(v_bio.tolist(), 101), (v_comp.tolist(), 202)]

        best_weights, best_fitness = mgr_bio.evolve_memory_niche_weights_py(
            orch_bio,
            golden_queries,
            generations=50,
            population_size=16,
            mutation_rate=0.25,
        )

        print(
            f"    - Pesos de Nicho Evolucionados: [Episodic: {best_weights[0]:.3f}, Documental: {best_weights[1]:.3f}, Conv: {best_weights[2]:.3f}]"
        )
        print(f"    - Fitness alcanzado: {best_fitness:.4f}")
        assert best_fitness >= 0.90

        # 5. Snapshot Híbrido y Gate de Promoción
        print("\n[*] 5. Sometiendo Época Híbrida al Gate de Promoción...")
        ep_hybrid = mgr_bio.create_snapshot_py(
            orch_bio, f"Híbrido Bio+Comp (Fitness: {best_fitness:.4f})", None
        )
        verdict_json = mgr_bio.evaluate_and_gate_py(ep_hybrid, golden_queries)
        verdict = json.loads(verdict_json)
        print("    - Veredicto Gate:")
        print(json.dumps(verdict, indent=4))

        assert verdict["passed"] is True
        assert verdict["needle_recall"] == 1.0
        assert verdict["action_taken"] == "PROMOTED_AND_SEALED"
        assert mgr_bio.active_epoch_id == ep_hybrid

        # 6. Validar Recuperación de ambas agujas en organismo Bio híbrido
        res_bio = orch_bio.retrieve_context_py(v_bio.tolist(), 1)
        res_comp = orch_bio.retrieve_context_py(v_comp.tolist(), 1)

        assert res_bio[0][1] == 101
        assert res_comp[0][1] == 202
        print(f"    - Aguja Bio Recuperada: ID {res_bio[0][1]} ('{res_bio[0][3]}')")
        print(f"    - Aguja Comp Recuperada: ID {res_comp[0][1]} ('{res_comp[0][3]}')")

        print("\n" + "=" * 80)
        print("🏆 CERTIFICACIÓN DE FASE 4 COMPLETADA AL 100%")
        print("   • Cross-Breeding de Memoria Inter-Organismos: VERIFICADO")
        print(
            f"   • Evolución Genética DNI de Nichos (Fitness {best_fitness:.4f}): CERTIFICADA"
        )
        print("   • Needle Recall 100% en Híbrido Multidisciplinario: CONFIRMADO")
        print("=" * 80)

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()


def test_memory_evolution_and_breeding():
    main()
