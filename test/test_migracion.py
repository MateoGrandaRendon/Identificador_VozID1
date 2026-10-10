"""
test/test_migracion.py
----------------------
Pruebas de migrar_sqlite_a_mongo.py con una base SQLite antigua de prueba
(en una carpeta temporal) y MongoDB en memoria (mongomock).
"""
import sqlite3
import uuid

import numpy as np
import pytest

import config
import database as db
import migrar_sqlite_a_mongo as migracion

pytestmark = pytest.mark.usefixtures("entorno")


def _base_legada(tmp_path, personas):
    """Crea una BD SQLite con el esquema antiguo; personas = {nombre: número de muestras}."""
    ruta_db = tmp_path / "voice_id.db"
    con = sqlite3.connect(ruta_db)
    con.execute("CREATE TABLE speakers (id INTEGER PRIMARY KEY, nombre TEXT, fecha_registro TEXT)")
    con.execute("CREATE TABLE voice_samples (speaker_id INTEGER, ruta_audio TEXT, ruta_embedding TEXT, "
                "fecha_creacion TEXT, version_embedding INTEGER)")
    for i, (nombre, n) in enumerate(personas.items(), start=1):
        con.execute("INSERT INTO speakers VALUES (?, ?, ?)", (i, nombre, "2026-01-01T10:00:00"))
        for _ in range(n):
            audio = config.AUDIO_DIR / f"{uuid.uuid4().hex}.wav.enc"
            emb = config.AUDIO_DIR / f"{uuid.uuid4().hex}.npy.enc"
            audio.write_bytes(b"x")
            emb.write_bytes(bytes(db._cifrar_embedding(np.random.rand(config.EMBEDDING_DIM))))
            con.execute("INSERT INTO voice_samples VALUES (?, ?, ?, ?, ?)",
                        (i, str(audio), str(emb), "2026-01-01T10:00:00", config.EMBEDDING_VERSION))
    con.commit()
    con.close()
    return ruta_db


def test_migra_nombres_validos_normalizados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_LEGADO_PATH", _base_legada(tmp_path, {"  Ana   María ": 4}))
    migracion.migrar(borrar_sqlite=False)
    speaker = db.obtener_speaker("ana maría")
    assert speaker["nombre"] == "Ana María" and len(speaker["muestras"]) == 4


@pytest.mark.parametrize("nombre", ["_usuario_prueba_", "<img src=x onerror=alert(1)>", "   ", "1Ana"])
def test_nombres_legados_no_validos_no_se_migran(tmp_path, monkeypatch, nombre):
    # "_usuario_prueba_" es el nombre reservado que la autoprueba de BD borra siempre (main.NOMBRE_PRUEBA).
    monkeypatch.setattr(config, "SQLITE_LEGADO_PATH", _base_legada(tmp_path, {nombre: 4}))
    migracion.migrar(borrar_sqlite=False)
    assert db.listar_speakers() == []
