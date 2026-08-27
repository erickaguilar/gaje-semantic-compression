#!/usr/bin/env python3
"""Test de endpoints REST de Épocas de Memoria y Consolidación en Web UI Server."""

import io
import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER_DIR = os.path.join(PROJECT_ROOT, "examples", "ui", "web_ui")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, SERVER_DIR)

from server import GajeHandler  # noqa: E402


class DummyServer:
    pass


class DummyGajeHandler(GajeHandler):
    def __init__(self, request_bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()
        self.client_address = ("127.0.0.1", 12345)
        self.server = DummyServer()
        self.raw_requestline = self.rfile.readline()
        self.parse_request()

    def handle(self):
        pass

    def finish(self):
        pass


class TestWebUIEpochEndpoints(unittest.TestCase):
    def test_01_get_epochs(self):
        req = b"GET /api/memory/epochs?organism=smollm2_adult&dim=576 HTTP/1.1\r\nHost: localhost\r\n\r\n"
        handler = DummyGajeHandler(req)
        handler.do_GET()
        response = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", response)
        self.assertIn('"status": "ok"', response)
        self.assertIn('"epochs"', response)
        print("✓ GET /api/memory/epochs respondio 200 OK con lista de epocas")

    def test_02_snapshot_and_rollback(self):
        # 1. Snapshot
        body = json.dumps(
            {"organism": "smollm2_adult", "comment": "Test Snapshot REST", "dim": 576}
        ).encode("utf-8")
        req = (
            f"POST /api/memory/epochs/snapshot HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode(
                "utf-8"
            )
            + body
        )
        handler = DummyGajeHandler(req)
        handler.do_POST()
        response = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", response)
        self.assertIn('"epoch_id"', response)
        print("✓ POST /api/memory/epochs/snapshot creo snapshot exitosamente")

        # 2. Rollback
        body_rb = json.dumps(
            {"organism": "smollm2_adult", "epoch_id": 1, "dim": 576}
        ).encode("utf-8")
        req_rb = (
            f"POST /api/memory/epochs/rollback HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body_rb)}\r\n\r\n".encode(
                "utf-8"
            )
            + body_rb
        )
        handler_rb = DummyGajeHandler(req_rb)
        handler_rb.do_POST()
        response_rb = handler_rb.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", response_rb)
        self.assertIn('"active_epoch_id": 1', response_rb)
        print("✓ POST /api/memory/epochs/rollback restauro epoca 1 exitosamente")

    def test_03_consolidate_sleep_cycle(self):
        body = json.dumps(
            {"organism": "smollm2_adult", "dedup_threshold": 0.95, "dim": 576}
        ).encode("utf-8")
        req = (
            f"POST /api/memory/epochs/consolidate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode(
                "utf-8"
            )
            + body
        )
        handler = DummyGajeHandler(req)
        handler.do_POST()
        response = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", response)
        self.assertIn('"stats"', response)
        print(
            "✓ POST /api/memory/epochs/consolidate ejecuto ciclo de sueno exitosamente"
        )


if __name__ == "__main__":
    unittest.main()
