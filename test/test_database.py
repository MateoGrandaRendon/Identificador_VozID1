"""
test/test_database.py
---------------------
Pruebas de la capa MongoDB (database.py / registro.py) y de las defensas de
seguridad, usando mongomock (MongoDB en memoria): no requieren un servidor
real, micrófono ni la clave del .env.
"""
import datetime
import uuid
from pathlib import Path

import numpy as np
import pytest

import config
import database as db
import security


pytestmark = pytest.mark.usefixtures("entorno")


def _muestras(n):
    """n pares (ruta_audio, embedding) con archivos reales (vacíos) en AUDIO_DIR."""
    pares = []
    for _ in range(n):
        ruta = config.AUDIO_DIR / f"{uuid.uuid4().hex}.wav.enc"
        ruta.write_bytes(b"x")
        pares.append((str(ruta), np.random.rand(config.EMBEDDING_DIM)))
    return pares


# ---------------------------------------------------------------------------
# Límite de 4 a 25 muestras por persona
# ---------------------------------------------------------------------------
def test_registro_exige_minimo_4_muestras():
    with pytest.raises(db.LimiteMuestrasError):
        db.registrar_speaker("Ana", _muestras(config.MIN_MUESTRAS_POR_PERSONA - 1))
    assert db.obtener_speaker("Ana") is None


def test_registro_rechaza_mas_de_25_muestras():
    with pytest.raises(db.LimiteMuestrasError):
        db.registrar_speaker("Ana", _muestras(config.MAX_MUESTRAS_POR_PERSONA + 1))


@pytest.mark.parametrize("n", [config.MIN_MUESTRAS_POR_PERSONA, config.MAX_MUESTRAS_POR_PERSONA])
def test_registro_acepta_los_extremos(n):
    assert db.registrar_speaker("Ana", _muestras(n)) == "ok"
    assert len(db.obtener_speaker("Ana")["muestras"]) == n


def test_agregar_muestras_no_supera_25():
    db.registrar_speaker("Ana", _muestras(20))
    assert db.agregar_muestras("Ana", _muestras(6)) == "limite"
    assert len(db.obtener_speaker("Ana")["muestras"]) == 20
    assert db.agregar_muestras("Ana", _muestras(5)) == "ok"
    assert len(db.obtener_speaker("Ana")["muestras"]) == 25
    assert db.agregar_muestras("Ana", _muestras(1)) == "limite"
    assert db.agregar_muestras("Nadie", _muestras(1)) == "no_existe"


def test_reemplazar_muestras_devuelve_audios_antiguos():
    viejas = _muestras(4)
    db.registrar_speaker("Ana", viejas)
    antiguos = db.reemplazar_muestras("Ana", _muestras(5))
    assert sorted(antiguos) == sorted(Path(ruta).name for ruta, _ in viejas)
    assert len(db.obtener_speaker("Ana")["muestras"]) == 5
    with pytest.raises(db.LimiteMuestrasError):
        db.reemplazar_muestras("Ana", _muestras(3))


def test_nombre_unico_sin_distinguir_mayusculas():
    assert db.registrar_speaker("Ana", _muestras(4)) == "ok"
    assert db.registrar_speaker("ANA", _muestras(4)) == "duplicado"
    db.registrar_speaker("Luis", _muestras(4))
    assert db.renombrar_speaker("luis", "ana") == "duplicado"
    assert db.renombrar_speaker("luis", "Luis Miguel") == "ok"


def test_embeddings_se_guardan_cifrados_y_se_recuperan():
    pares = _muestras(4)
    db.registrar_speaker("Ana", pares)
    doc = db._speakers().find_one({"nombre_clave": "ana"})
    crudo = bytes(doc["muestras"][0]["embedding"])
    assert b"NUMPY" not in crudo  # en la BD no hay un .npy en claro
    recuperados = db.obtener_embeddings_todos()
    assert len(recuperados) == 4
    np.testing.assert_allclose(recuperados[0][1], pares[0][1])


def test_eliminar_borra_documento_y_audios():
    pares = _muestras(4)
    db.registrar_speaker("Ana", pares)
    existia, borrados = db.eliminar_speaker_completo("ana")
    assert existia and borrados == 4
    assert not any(config.AUDIO_DIR.iterdir())
    assert db.eliminar_speaker_completo("Ana") == (False, 0)


def test_listar_no_descifra_y_detecta_desactualizados():
    db.registrar_speaker("Ana", _muestras(4))
    doc = db._speakers().find_one({"nombre_clave": "ana"})
    for m in doc["muestras"]:
        m["version_embedding"] = config.EMBEDDING_VERSION - 1  # como tras subir la versión del motor
    db._speakers().replace_one({"_id": doc["_id"]}, doc)
    [p] = db.listar_speakers()
    assert p["total_muestras"] == 4 and p["necesita_reentrenar"]


# ---------------------------------------------------------------------------
# Inyección NoSQL, recorrido de directorios y validación de entradas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("malicioso", [{"$ne": ""}, {"$gt": ""}, ["Ana"], 1, None])
def test_inyeccion_nosql_rechazada(malicioso):
    db.registrar_speaker("Ana", _muestras(4))
    with pytest.raises(db.BaseDatosError):
        db.obtener_speaker(malicioso)
    with pytest.raises(db.BaseDatosError):
        db.eliminar_speaker_completo(malicioso)
    with pytest.raises(ValueError):
        security.validar_nombre_persona(malicioso)
    assert db.obtener_speaker("Ana") is not None


@pytest.mark.parametrize("archivo", ["../config.py", "..\\..\\x.wav.enc", "/etc/passwd", "a.wav.enc", "C:/x.wav.enc"])
def test_path_traversal_rechazado(archivo):
    with pytest.raises(db.BaseDatosError):
        db.ruta_audio_segura(archivo)


@pytest.mark.parametrize("nombre", ["<script>alert(1)</script>", "_usuario_prueba_", "a" * 61, "   ", "Ana; rm -rf /", "$where"])
def test_nombres_invalidos(nombre):
    with pytest.raises(ValueError):
        security.validar_nombre_persona(nombre)


def test_nombre_valido_se_normaliza():
    assert security.validar_nombre_persona("  José   María O'Neil-Pérez ") == "José María O'Neil-Pérez"


def test_uri_remota_sin_tls_rechazada():
    with pytest.raises(db.BaseDatosError):
        db._validar_uri("mongodb://usuario:clave@db.example.com:27017")
    with pytest.raises(db.BaseDatosError):
        db._validar_uri("mongodb://db.example.com/?tls=true&tlsAllowInvalidCertificates=true")
    db._validar_uri("mongodb://db.example.com/?tls=true")
    db._validar_uri("mongodb://127.0.0.1:27017")


# ---------------------------------------------------------------------------
# PIN: fuerza bruta con bloqueo progresivo y persistente
# ---------------------------------------------------------------------------
def test_bloqueo_progresivo_y_persistente():
    control = security.ControlAcceso()
    for _ in range(security.MAX_INTENTOS - 1):
        assert control.intentar("000000")[0] == "incorrecto"
    resultado, segundos = control.intentar("000000")
    assert resultado == "bloqueado" and segundos == security.BLOQUEO_SEGUNDOS
    # Una instancia nueva (= reiniciar el programa) sigue bloqueada, incluso con el PIN correcto.
    assert security.ControlAcceso().intentar("147258")[0] == "bloqueado"

    # Al expirar, el siguiente bloqueo dura el doble.
    db.guardar_estado_pin(0, 1, datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1))
    for _ in range(security.MAX_INTENTOS):
        resultado, segundos = control.intentar("000000")
    assert resultado == "bloqueado" and segundos == 2 * security.BLOQUEO_SEGUNDOS


def test_pin_correcto_reinicia_contador():
    control = security.ControlAcceso()
    control.intentar("000000")
    assert control.intentar("147258") == ("ok", 0)
    assert db.leer_estado_pin()["fallos"] == 0


@pytest.mark.parametrize("pin", ["12345", "111111", "123456", "987654"])
def test_pin_debil_rechazado(pin):
    with pytest.raises(ValueError):
        security.validar_pin_nuevo(pin)


def test_verificar_pin_rechaza_tipos_y_longitudes_raras():
    assert not security.verificar_pin({"$ne": ""}, "sal", config.PIN_HASH)
    assert not security.verificar_pin("1" * 10_000, "sal", config.PIN_HASH)
