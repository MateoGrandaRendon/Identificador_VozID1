"""
test/test_robustez.py
---------------------
Pruebas de QA «intentando romper la app»: audio corrupto o corto, nombres que
crecen al normalizarse, llamadas fuera de orden o repetidas desde la GUI,
doble clic concurrente, clave o .env mal copiados, MongoDB caído a mitad de
sesión y cortes de teclado o consola sin UTF-8 en el CLI.
"""
import io
import sys
import threading
import time
import uuid

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pymongo.errors import ServerSelectionTimeoutError

import audio_processor as ap
import config
import database as db
import main
import security
from frontend import bridge
from test.conftest import PIN_PRUEBA

pytestmark = pytest.mark.usefixtures("entorno")
SR = config.SAMPLE_RATE


def _voz(segundos):
    """Señal sintética tipo voz (tono modulado con pausas + ruido bajo) que pasa la validación."""
    t = np.arange(int(segundos * SR)) / SR
    envolvente = (np.sin(2 * np.pi * 2 * t) > 0).astype(float)
    ruido = 0.002 * np.random.default_rng(0).standard_normal(len(t))
    return (0.3 * envolvente * np.sin(2 * np.pi * 150 * t) + ruido).astype(np.float32)


@pytest.fixture
def sesion(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(bridge.ap, "extraer_embedding", lambda audio: np.random.rand(config.EMBEDDING_DIM))
    api = bridge.Api()
    assert api.login(PIN_PRUEBA)["ok"]
    return api


# ---------------------------------------------------------------------------
# Audio corrupto: antes pasaba la validación y rompía extraer_embedding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("audio", [
    np.full(SR * 3, np.nan, dtype=np.float32),
    np.where(np.arange(SR * 3) % 1000 == 0, np.nan, _voz(3)).astype(np.float32),
    np.concatenate([_voz(3), np.array([np.inf], dtype=np.float32)]),
    np.stack([_voz(3), _voz(3)], axis=1),          # estéreo
    (_voz(3) * 32767).astype(np.int16),           # enteros en vez de float
    _voz(0.05),                                   # demasiado corto para el embedding
])
def test_audio_invalido_se_rechaza_antes_del_embedding(audio):
    valida, motivo = ap.validar_calidad_muestra(audio)
    assert valida is False and motivo


def test_audio_de_voz_normal_sigue_siendo_valido():
    assert ap.validar_calidad_muestra(_voz(3)) == (True, "")


# ---------------------------------------------------------------------------
# Nombres que crecen con casefold ("ß" -> "ss")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nombre", ["ß" * 31, "A" + "ß" * (config.NOMBRE_MAX_LARGO - 1)])
def test_nombre_valido_que_crece_al_normalizar_se_puede_registrar(nombre):
    nombre = security.validar_nombre_persona(nombre)
    muestras = []
    for _ in range(config.MIN_MUESTRAS_POR_PERSONA):
        ruta = config.AUDIO_DIR / f"{uuid.uuid4().hex}.wav.enc"
        ruta.write_bytes(b"x")
        muestras.append((str(ruta), np.random.rand(config.EMBEDDING_DIM)))
    assert db.registrar_speaker(nombre, muestras) == "ok"
    assert db.obtener_speaker(nombre)["nombre"] == nombre


# ---------------------------------------------------------------------------
# GUI: llamadas fuera de orden / repetidas
# ---------------------------------------------------------------------------
def test_detener_dos_veces_no_reutiliza_la_grabacion_anterior(sesion):
    sesion._rec.active = True
    sesion._rec.procesar(_voz(0.1))
    assert sesion.record_stop()["ok"]
    r = sesion.record_stop()
    assert r["code"] == "no_audio" and sesion._pending is None


def test_detener_sin_haber_iniciado(sesion):
    assert sesion.record_stop()["code"] == "no_audio"
    assert sesion.analysis_stop()["code"] == "no_audio"


def test_doble_clic_en_completar_no_duplica_muestras(sesion, monkeypatch):
    sesion.wizard_start("Ana", "registro")
    for _ in range(4):
        sesion._pending = np.zeros(10, dtype=np.float32)
        sesion.wizard_keep()
    assert sesion.wizard_finish()["ok"]
    sesion.wizard_start("Ana", "ampliar")
    for _ in range(2):
        sesion._pending = np.zeros(10, dtype=np.float32)
        sesion.wizard_keep()

    dentro, seguir = threading.Event(), threading.Event()
    original = bridge.registro.guardar_perfil

    def lento(*args):
        dentro.set()
        seguir.wait(5)
        return original(*args)
    monkeypatch.setattr(bridge.registro, "guardar_perfil", lento)

    resultados = []
    primero = threading.Thread(target=lambda: resultados.append(sesion.wizard_finish()))
    primero.start()
    assert dentro.wait(5)
    segundo = sesion.wizard_finish()          # el doble clic llega mientras se guarda
    seguir.set()
    primero.join()
    assert resultados[0]["ok"] and segundo["code"] == "busy"
    assert len(db.obtener_speaker("Ana")["muestras"]) == 6


# ---------------------------------------------------------------------------
# Configuración mal copiada
# ---------------------------------------------------------------------------
def test_clave_fernet_con_formato_invalido_se_detecta_al_arrancar(monkeypatch):
    monkeypatch.setattr(config, "FERNET_KEY", "clave-mal-copiada")
    with pytest.raises(RuntimeError, match="VOICE_ID_FERNET_KEY"):
        security.verificar_clave_cifrado()


def test_clave_distinta_no_dice_que_no_hay_personas(sesion, monkeypatch):
    sesion.wizard_start("Ana", "registro")
    for _ in range(4):
        sesion._pending = np.zeros(10, dtype=np.float32)
        sesion.wizard_keep()
    assert sesion.wizard_finish()["ok"]
    monkeypatch.setattr(config, "FERNET_KEY", Fernet.generate_key().decode())   # otra clave válida
    assert sesion.identify_info()["code"] == "key"


@pytest.mark.parametrize("valor,esperado", [("", 5000), ("  ", 5000), ("7000", 7000)])
def test_entero_del_env_vacio_usa_el_valor_por_defecto(monkeypatch, valor, esperado):
    monkeypatch.setenv("VOICE_ID_PRUEBA_ENTERO", valor)
    assert config._entero_env("VOICE_ID_PRUEBA_ENTERO", 5000) == esperado


def test_entero_del_env_no_numerico_da_mensaje_claro(monkeypatch):
    monkeypatch.setenv("VOICE_ID_PRUEBA_ENTERO", "cinco mil")
    with pytest.raises(SystemExit, match="VOICE_ID_PRUEBA_ENTERO"):
        config._entero_env("VOICE_ID_PRUEBA_ENTERO", 5000)


# ---------------------------------------------------------------------------
# MongoDB cae a mitad de sesión
# ---------------------------------------------------------------------------
def test_mongodb_caido_da_un_mensaje_util(sesion, monkeypatch):
    def caido():
        raise ServerSelectionTimeoutError("127.0.0.1:27017: timed out")
    monkeypatch.setattr(db, "_speakers", caido)
    r = sesion.list_speakers()
    assert r["code"] == "db" and "MongoDB" in r["error"]


# ---------------------------------------------------------------------------
# CLI: Ctrl+C / Ctrl+Z en el PIN y consola sin UTF-8
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corte", [KeyboardInterrupt, EOFError])
def test_cli_cortado_en_el_pin_sale_sin_traza(monkeypatch, capsys, corte):
    def interrumpir():
        raise corte
    monkeypatch.setattr(main.security, "configurar_logs", lambda: None)
    monkeypatch.setattr(main.security, "pedir_acceso", interrumpir)
    main._arrancar()
    assert "Hasta luego" in capsys.readouterr().out


def test_cli_con_clave_invalida_sale_con_mensaje(monkeypatch, capsys):
    monkeypatch.setattr(main.security, "configurar_logs", lambda: None)
    monkeypatch.setattr(config, "FERNET_KEY", "clave-mal-copiada")
    with pytest.raises(SystemExit):
        main._arrancar()
    assert "VOICE_ID_FERNET_KEY" in capsys.readouterr().out


def test_consola_cp1252_no_rompe_con_simbolos(monkeypatch):
    salida = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", salida)
    main.configurar_consola()
    print("SNR ≈ 3 dB ⚙️ ✅")       # antes: UnicodeEncodeError
