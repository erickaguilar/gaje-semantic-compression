#!/usr/bin/env python3
"""
🧬 GAJE HELIX: CERTIFICACIÓN DE FASE 2 — GATE DE PROMOCIÓN Y COMANDOS CLI (gaje-cli epoch)
================================================================================
Valida los subcomandos de `gaje-cli epoch` (list, snapshot, rollback, promote, seal, evaluate)
y el funcionamiento automatizado del Gate de Promoción (evaluación con consultas doradas,
promoción y sellado atómico, y rechazo con rollback automático al padre ante degradación).
================================================================================
"""

import json
import shutil
import tempfile
import subprocess
import numpy as np

import gaje.core._impl as _impl


def run_cli(args):
    cmd = ["./target/debug/gaje-cli", "epoch"] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Error ejecutando CLI {' '.join(cmd)}:\n{res.stderr}")
    return res.stdout


def main():
    print(
        "================================================================================"
    )
    print("🧬 GAJE HELIX: CERTIFICACIÓN DE FASE 2 — GATE DE PROMOCIÓN Y CLI")
    print(
        "================================================================================"
    )

    temp_root = tempfile.mkdtemp(prefix="gaje_phase2_cert_")
    organism = "smollm2_gate_cert"
    dim = 64

    try:
        # 1. Validar gaje-cli epoch list inicial
        print("\n[*] 1. Validando `gaje-cli epoch list`...")
        out_list = run_cli(
            ["list", "--organism", organism, "--root", temp_root, "--dim", str(dim)]
        )
        print(out_list)
        assert "Época Activa: 1" in out_list
        assert "*1" in out_list

        # 2. Validar gaje-cli epoch snapshot
        print("\n[*] 2. Validando `gaje-cli epoch snapshot`...")
        out_snap = run_cli(
            [
                "snapshot",
                "--organism",
                organism,
                "--root",
                temp_root,
                "--dim",
                str(dim),
                "--comment",
                "Snapshot Test CLI",
            ]
        )
        print(out_snap)
        assert "Época ID 2" in out_snap

        out_list2 = run_cli(
            ["list", "--organism", organism, "--root", temp_root, "--dim", str(dim)]
        )
        assert "*2" in out_list2

        # 3. Validar gaje-cli epoch rollback
        print("\n[*] 3. Validando `gaje-cli epoch rollback` a época 1...")
        out_rb = run_cli(
            [
                "rollback",
                "--organism",
                organism,
                "--root",
                temp_root,
                "--dim",
                str(dim),
                "--epoch",
                "1",
            ]
        )
        print(out_rb)
        assert "Época activa ahora es ID 1" in out_rb

        out_list3 = run_cli(
            ["list", "--organism", organism, "--root", temp_root, "--dim", str(dim)]
        )
        assert "*1" in out_list3

        # 4. Validar gaje-cli epoch promote y seal
        print("\n[*] 4. Validando `gaje-cli epoch promote` y `seal`...")
        run_cli(
            [
                "promote",
                "--organism",
                organism,
                "--root",
                temp_root,
                "--dim",
                str(dim),
                "--epoch",
                "2",
            ]
        )
        run_cli(
            [
                "seal",
                "--organism",
                organism,
                "--root",
                temp_root,
                "--dim",
                str(dim),
                "--epoch",
                "2",
            ]
        )
        out_list4 = run_cli(
            ["list", "--organism", organism, "--root", temp_root, "--dim", str(dim)]
        )
        print(out_list4)
        assert "SEALED" in out_list4

        # 5. Validar Gate de Promoción con Python (Caso Éxito 100% Recall)
        print("\n[*] 5. Validando Gate de Promoción (Caso Éxito: 100% Recall)...")
        mgr = _impl.EpochManager(temp_root, organism, dim)
        orch = _impl.IslandOrchestrator(dim)

        # Agregamos aguja dorada
        golden_vec = np.random.randn(dim).astype(np.float32)
        golden_vec /= np.linalg.norm(golden_vec)
        orch.add_memory_py(
            "documental", 777, golden_vec.tolist(), "Aguja Dorada: Token Secreto 777"
        )

        ep3_id = mgr.create_snapshot_py(orch, "Época Candidata con Aguja Dorada", None)
        print(f"    - Creada Época Candidata {ep3_id}")

        golden_queries = [(golden_vec.tolist(), 777)]
        verdict_str = mgr.evaluate_and_gate_py(ep3_id, golden_queries)
        verdict = json.loads(verdict_str)

        print("    - Veredicto Gate:")
        print(json.dumps(verdict, indent=2))

        assert verdict["passed"] is True
        assert verdict["needle_recall"] == 1.0
        assert verdict["action_taken"] == "PROMOTED_AND_SEALED"
        assert mgr.active_epoch_id == ep3_id

        # 6. Validar Gate de Promoción (Caso Fallo / Degradación: 0% Recall -> Rollback)
        print(
            "\n[*] 6. Validando Gate de Promoción (Caso Fallo: 0% Recall -> Rollback Automático)..."
        )
        # Ingesta ruidosa sin la aguja esperada
        orch_corrupt = _impl.IslandOrchestrator(dim)
        v_noise = np.random.randn(dim).astype(np.float32)
        v_noise /= np.linalg.norm(v_noise)
        orch_corrupt.add_memory_py("episodic", 999, v_noise.tolist(), "Ruido Corrupto")

        ep4_id = mgr.create_snapshot_py(orch_corrupt, "Época Candidata Corrupta", None)
        print(f"    - Creada Época Candidata Corrupta {ep4_id}")

        # Evaluamos con la aguja esperada (no presente en ep4)
        verdict_fail_str = mgr.evaluate_and_gate_py(ep4_id, golden_queries)
        verdict_fail = json.loads(verdict_fail_str)

        print("    - Veredicto Gate (Degradación detectada):")
        print(json.dumps(verdict_fail, indent=2))

        assert verdict_fail["passed"] is False
        assert verdict_fail["needle_recall"] == 0.0
        assert "ROLLBACK" in verdict_fail["action_taken"]
        assert (
            mgr.active_epoch_id == ep3_id
        )  # Rollback automático a la época canónica previa

        # 7. Auditoría final de linaje
        manifests = json.loads(mgr.list_epochs_py())
        print(f"\n[*] 7. Total de épocas en linaje: {len(manifests)}")
        print(f"    - Época Activa: {mgr.active_epoch_id}")
        assert manifests[-1]["verdict"] == "REJECTED"

        print("\n" + "=" * 80)
        print("🏆 CERTIFICACIÓN DE FASE 2 COMPLETADA AL 100%")
        print(
            "   • Subcomandos `gaje-cli epoch` (list, snapshot, rollback, promote, seal, evaluate): VERIFICADOS"
        )
        print(
            "   • Gate de Promoción Automático (Promoción y Sellado en éxito / Rollback en fallo): CERTIFICADO"
        )
        print("=" * 80)

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()


def test_epoch_cli_and_gate():
    main()
