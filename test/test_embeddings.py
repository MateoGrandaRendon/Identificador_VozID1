"""
tests/test_embeddings.py
-------------------------
Pruebas de regresión para el bug crítico:

    ValueError: shapes (156,) and (60,) not aligned: 156 (dim 0) != 60 (dim 0)

Objetivo: que si en el futuro alguien cambia N_MFCC, agrega/quita un
bloque de características en extraer_embedding, o toca similitud_coseno
/ distancia_euclidiana, el pipeline falle EN `pytest` — antes de llegar
a producción — nunca en medio de una identificación real del usuario.

Ejecutar desde la raíz del proyecto (con el .venv activado):
    pytest -v
"""
import numpy as np
import pytest

import config
import audio_processor as ap


def _audio_sintetico(segundos: float, freq_hz: float = 180.0) -> np.ndarray:
    """Tono senoidal simple; no requiere micrófono para correr en CI."""
    sr = config.SAMPLE_RATE
    t = np.arange(int(segundos * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


@pytest.mark.parametrize("segundos,freq", [
    (0.3, 150.0),    # muestra muy corta (límite inferior realista)
    (1.0, 200.0),
    (4.0, 120.0),
    (10.0, 250.0),
    (15.0, 90.0),     # muestra larga (REGISTRO_DURACION_MAX)
])
def test_extraer_embedding_dimension_fija(segundos, freq):
    """El embedding SIEMPRE debe tener config.EMBEDDING_DIM componentes,
    sin importar la duración o el contenido del audio de entrada."""
    audio = _audio_sintetico(segundos, freq)
    embedding = ap.extraer_embedding(audio)
    assert embedding.shape == (config.EMBEDDING_DIM,)
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()


def test_extraer_embedding_silencio_no_crashea():
    """Silencio total no debe producir NaN/Inf ni una dimensión distinta
    (los bloques de formantes/temporal usan zeros(3)/zeros(4) como
    fallback cuando no hay frames con voz detectada)."""
    audio = np.zeros(int(1.0 * config.SAMPLE_RATE), dtype=np.float32)
    embedding = ap.extraer_embedding(audio)
    assert embedding.shape == (config.EMBEDDING_DIM,)
    assert not np.isnan(embedding).any()


def test_embedding_dim_coincide_con_config():
    """Guardarraíl de configuración: si alguien cambia N_MFCC u otro
    parámetro del pipeline sin actualizar config.EMBEDDING_DIM (y subir
    config.EMBEDDING_VERSION), esta prueba debe fallar de inmediato — esa
    es la forma correcta de detectar el desfase EN DESARROLLO, no cuando
    un usuario real intente identificarse."""
    audio = _audio_sintetico(3.0)
    embedding = ap.extraer_embedding(audio)
    assert embedding.shape[0] == config.EMBEDDING_DIM, (
        f"config.EMBEDDING_DIM={config.EMBEDDING_DIM} pero extraer_embedding "
        f"produjo {embedding.shape[0]} dimensiones. Si el cambio es intencional: "
        f"actualiza EMBEDDING_DIM y sube EMBEDDING_VERSION en config.py."
    )


def test_similitud_coseno_misma_persona_alta():
    audio = _audio_sintetico(2.0)
    emb1 = ap.extraer_embedding(audio)
    emb2 = ap.extraer_embedding(audio)
    assert ap.similitud_coseno(emb1, emb2) == pytest.approx(1.0, abs=1e-6)


def test_similitud_coseno_dimensiones_distintas_no_crashea():
    """Reproduce EXACTAMENTE el escenario del traceback original:
    (156,) vs (60,). Antes lanzaba ValueError; ahora debe devolver -1.0
    ('lo más distinto posible') sin propagar ninguna excepción."""
    emb_nuevo = ap.extraer_embedding(_audio_sintetico(2.0))
    emb_viejo = np.random.randn(60)
    assert emb_nuevo.shape != emb_viejo.shape
    assert ap.similitud_coseno(emb_nuevo, emb_viejo) == -1.0


def test_distancia_euclidiana_dimensiones_distintas_no_crashea():
    emb_nuevo = ap.extraer_embedding(_audio_sintetico(2.0))
    emb_viejo = np.random.randn(60)
    assert ap.distancia_euclidiana(emb_nuevo, emb_viejo) == float("inf")


def test_similitud_coseno_vector_cero_no_crashea():
    """Un embedding todo-ceros (norma 0) no debe causar ZeroDivisionError."""
    audio = _audio_sintetico(2.0)
    emb = ap.extraer_embedding(audio)
    cero = np.zeros_like(emb)
    assert ap.similitud_coseno(emb, cero) == 0.0
