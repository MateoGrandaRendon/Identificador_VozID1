"""
test/test_motor_ecapa.py
------------------------
Motor de embeddings ECAPA (motor_ecapa.py) y recálculo de embeddings desde el
audio guardado (recalcular_embeddings.py). Las pruebas que necesitan el modelo
real se saltan si no está descargado (python -m motor_ecapa).
"""
import io
import shutil
import uuid

import numpy as np
import pytest
import soundfile as sf

import config
import database as db
import motor_ecapa
import recalcular_embeddings
import security

pytestmark = pytest.mark.usefixtures("entorno")
MODELO_REAL = config.MODELO_DIR / motor_ecapa.ARCHIVO
requiere_modelo = pytest.mark.skipif(not MODELO_REAL.is_file(),
                                     reason="modelo ECAPA no descargado (python -m motor_ecapa)")


def _voz(f0=180.0, segundos=3.0, semilla=0):
    rng = np.random.default_rng(semilla)
    t = np.arange(int(segundos * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    señal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * (np.sin(2 * np.pi * 3 * t) > -0.3)
    return (0.1 * señal + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def modelo_aislado(tmp_path, monkeypatch):
    """Carpeta de modelo propia y motor sin cargar (para probar la verificación del archivo)."""
    monkeypatch.setattr(config, "MODELO_DIR", tmp_path / "modelo")
    monkeypatch.setattr(motor_ecapa, "_modelo", None)
    monkeypatch.setattr(motor_ecapa, "_fbank", None)
    return tmp_path / "modelo"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
@requiere_modelo
def test_embedding_ecapa_tiene_192_dimensiones_y_norma_1():
    e = motor_ecapa.extraer_embedding(_voz())
    assert e.shape == (192,) and not np.isnan(e).any()
    assert np.linalg.norm(e) == pytest.approx(1.0)


@requiere_modelo
def test_embedding_ecapa_es_determinista():
    audio = _voz()
    np.testing.assert_allclose(motor_ecapa.extraer_embedding(audio), motor_ecapa.extraer_embedding(audio))


@requiere_modelo
def test_embedding_ecapa_remuestrea_otras_frecuencias():
    import librosa
    audio_44k = librosa.resample(_voz(), orig_sr=16000, target_sr=44100)
    e16 = motor_ecapa.extraer_embedding(_voz())
    e44 = motor_ecapa.extraer_embedding(audio_44k, samplerate=44100)
    assert float(np.dot(e16, e44)) > 0.95


@requiere_modelo
def test_silencio_no_produce_nan():
    e = motor_ecapa.extraer_embedding(np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32))
    assert not np.isnan(e).any()


# ---------------------------------------------------------------------------
# Verificación del archivo del modelo
# ---------------------------------------------------------------------------
def test_sin_modelo_da_instrucciones_claras(modelo_aislado):
    with pytest.raises(motor_ecapa.MotorNoDisponible, match="python -m motor_ecapa"):
        motor_ecapa.verificar()


@requiere_modelo
def test_modelo_alterado_se_rechaza_antes_de_cargarlo(modelo_aislado):
    modelo_aislado.mkdir(parents=True)
    copia = modelo_aislado / motor_ecapa.ARCHIVO
    shutil.copyfile(MODELO_REAL, copia)
    with open(copia, "r+b") as f:          # un solo byte distinto
        f.seek(1000)
        byte = f.read(1)
        f.seek(1000)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(motor_ecapa.MotorNoDisponible, match="no coincide"):
        motor_ecapa.verificar()
    assert motor_ecapa._modelo is None


# ---------------------------------------------------------------------------
# recalcular_embeddings.py: cambio de motor sin volver a grabar
# ---------------------------------------------------------------------------
def _perfil_antiguo(nombre, con_audio=True):
    """Persona con embeddings del motor anterior (versión 2, 156 dim) y su audio en AUDIO_DIR."""
    muestras = []
    for i in range(4):
        archivo = f"{uuid.uuid4().hex}.wav.enc"
        if con_audio:
            buffer = io.BytesIO()
            sf.write(buffer, _voz(semilla=i), config.SAMPLE_RATE, format="WAV")
            db.ruta_audio_segura(archivo).write_bytes(security.encriptar_bytes(buffer.getvalue()))
        muestras.append({"archivo_audio": archivo,
                         "embedding": db._cifrar_embedding(np.random.rand(156)),
                         "version_embedding": 2, "dim": 156, "fecha_creacion": db._ahora()})
    db._speakers().insert_one({"nombre": nombre, "nombre_clave": nombre.casefold(),
                               "fecha_registro": db._ahora(), "muestras": muestras})


@requiere_modelo
def test_recalcular_actualiza_al_motor_actual_desde_el_audio(monkeypatch):
    monkeypatch.setattr(config, "MOTOR_EMBEDDING", "ecapa")
    monkeypatch.setattr(config, "EMBEDDING_VERSION", 3)
    monkeypatch.setattr(config, "EMBEDDING_DIM", 192)
    _perfil_antiguo("Ana")
    _perfil_antiguo("Luis", con_audio=False)
    resumen = recalcular_embeddings.recalcular()
    assert resumen["recalculadas"] == 4 and set(resumen["sin_audio"]) == {"Luis"}
    [ana] = [p for p in db.listar_speakers() if p["nombre"] == "Ana"]
    assert ana["muestras_compatibles"] == 4 and not ana["necesita_reentrenar"]
    assert recalcular_embeddings.recalcular()["recalculadas"] == 0      # idempotente
