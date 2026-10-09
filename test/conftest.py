"""Fixtures compartidas: MongoDB en memoria (mongomock), clave Fernet y PIN de prueba."""
import mongomock
import pytest
from cryptography.fernet import Fernet

import config
import database as db
import security

PIN_PRUEBA = "147258"


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FERNET_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(config, "PIN_SALT", "sal")
    monkeypatch.setattr(config, "PIN_HASH", security._hash_pin(PIN_PRUEBA, "sal", 1000))
    monkeypatch.setattr(config, "PIN_ITERACIONES", 1000)
    db.usar_cliente(mongomock.MongoClient())
    db.init_db()
    yield
    db.usar_cliente(None)
