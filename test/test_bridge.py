"""
test/test_bridge.py
-------------------
Pruebas de la API que pywebview expone a JavaScript (frontend/bridge.py):
autenticación, sesión, rate limiting, validación de entradas, límite de
muestras del asistente y que nunca se filtren mensajes internos.
"""
import time

import numpy as np
import pytest

import config
import database as db
from frontend import bridge
from test.conftest import PIN_PRUEBA

pytestmark = pytest.mark.usefixtures("entorno")

PUBLICOS = {
    "get_config", "auth_status", "login", "logout", "record_start", "get_live", "record_stop",
    "record_cancel", "list_speakers", "rename_speaker", "delete_speaker", "wizard_start",
    "wizard_keep", "wizard_finish", "wizard_cancel", "identify_info", "identify_run",
    "analysis_start", "analysis_stop", "run_test", "exit",
}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)  # login espera 1 s tras un PIN incorrecto
    monkeypatch.setattr(bridge.ap, "extraer_embedding", lambda audio: np.random.rand(config.EMBEDDING_DIM))
    return bridge.Api()


@pytest.fixture
def sesion(api):
    assert api.login(PIN_PRUEBA)["ok"]
    return api


def test_superficie_publica_es_exactamente_la_esperada():
    publicos = {n for n in dir(bridge.Api) if not n.startswith("_") and callable(getattr(bridge.Api, n))}
    assert publicos == PUBLICOS


def test_metodos_protegidos_exigen_sesion(api):
    for metodo in ("list_speakers", "delete_speaker", "record_start", "run_test"):
        r = getattr(api, metodo)(*(["Ana"] if metodo == "delete_speaker" else [1] if metodo == "run_test" else []))
        assert r["code"] == "auth", metodo


def test_login_rechaza_tipos_no_texto(api):
    assert api.login({"$ne": ""})["code"] == "bad_pin"
    assert api.login("000000")["code"] == "bad_pin"
    assert api.auth_status()["data"]["authenticated"] is False


def test_sesion_caduca_por_inactividad_y_descarta_datos(sesion):
    sesion._pending = np.zeros(10)
    sesion._ultima_actividad -= config.SESION_INACTIVIDAD_SEG + 1
    assert sesion.list_speakers()["code"] == "auth"
    assert sesion._pending is None and sesion._auth is False


def test_sesion_caduca_por_tiempo_maximo(sesion):
    sesion._inicio_sesion -= config.SESION_MAX_SEG + 1
    assert sesion.list_speakers()["code"] == "auth"


def test_logout(sesion):
    assert sesion.logout()["ok"]
    assert sesion.list_speakers()["code"] == "auth"


def test_rate_limiting(api):
    maximo, _ = bridge._LIMITE_DEFECTO
    respuestas = [api.get_config() for _ in range(maximo + 1)]
    assert all(r["ok"] for r in respuestas[:maximo])
    assert respuestas[-1]["code"] == "rate_limited"


@pytest.mark.parametrize("n", ["1", True, 4, None, {"$gt": 0}])
def test_run_test_valida_parametro(sesion, n):
    assert sesion.run_test(n)["code"] == "invalid"


@pytest.mark.parametrize("nombre,modo", [({"$ne": ""}, "registro"), ("<img src=x onerror=alert(1)>", "registro"),
                                         ("Ana", "admin"), ("Ana", None)])
def test_wizard_start_valida_entradas(sesion, nombre, modo):
    assert sesion.wizard_start(nombre, modo)["code"] == "invalid"


def test_argumentos_de_mas_no_rompen_el_puente(sesion):
    r = sesion.logout("extra", "args")
    assert r["ok"] is False and r["code"] == "invalid"


def test_error_interno_no_expone_detalles(sesion, monkeypatch):
    def falla():
        raise RuntimeError("mongodb://admin:secreto@10.0.0.5 C:\\ruta\\interna")
    monkeypatch.setattr(bridge.db, "listar_speakers", falla)
    r = sesion.list_speakers()
    assert r["code"] == "backend_error"
    assert "secreto" not in r["error"] and "ruta" not in r["error"]


def _grabar(api, n):
    for _ in range(n):
        api._pending = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        r = api.wizard_keep()
        assert r["ok"], r
    return r["data"]


def test_asistente_registro_entre_4_y_25(sesion):
    info = sesion.wizard_start("Ana", "registro")["data"]
    assert (info["min"], info["max"]) == (config.MIN_MUESTRAS_POR_PERSONA, config.MAX_MUESTRAS_POR_PERSONA)
    _grabar(sesion, config.MIN_MUESTRAS_POR_PERSONA - 1)
    assert sesion.wizard_finish()["code"] == "incomplete"
    info = _grabar(sesion, config.MAX_MUESTRAS_POR_PERSONA - config.MIN_MUESTRAS_POR_PERSONA + 1)
    assert info["lleno"] and info["guardadas"] == config.MAX_MUESTRAS_POR_PERSONA
    sesion._pending = np.zeros(10, dtype=np.float32)
    assert sesion.wizard_keep()["code"] == "limit"
    r = sesion.wizard_finish()
    assert r["ok"] and r["data"]["guardadas"] == config.MAX_MUESTRAS_POR_PERSONA
    assert len(db.obtener_speaker("Ana")["muestras"]) == config.MAX_MUESTRAS_POR_PERSONA


def test_asistente_ampliar_respeta_el_total(sesion):
    sesion.wizard_start("Ana", "registro")
    _grabar(sesion, 22)
    assert sesion.wizard_finish()["ok"]

    info = sesion.wizard_start("ana", "ampliar")["data"]
    assert (info["min"], info["max"], info["existentes"]) == (1, 3, 22)
    _grabar(sesion, 3)
    assert sesion.wizard_finish()["ok"]
    assert sesion.wizard_start("Ana", "ampliar")["code"] == "limit"


def test_registro_duplicado_y_persona_inexistente(sesion):
    sesion.wizard_start("Ana", "registro")
    _grabar(sesion, 4)
    sesion.wizard_finish()
    assert sesion.wizard_start("ANA", "registro")["code"] == "duplicate"
    assert sesion.wizard_start("Nadie", "reentrenar")["code"] == "not_found"


def test_tope_de_grabacion_en_memoria():
    rec = bridge._Recorder()
    bloque = np.zeros(bridge._BLOQUE, dtype=np.float32)
    for _ in range(bridge._MAX_MUESTRAS_AUDIO // bridge._BLOQUE + 50):
        rec.procesar(bloque)
    assert rec.total <= bridge._MAX_MUESTRAS_AUDIO
