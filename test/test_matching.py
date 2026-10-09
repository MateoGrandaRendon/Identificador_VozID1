"""
test/test_matching.py
----------------------
Pruebas de la lógica de decisión en matching.py: filtrado de perfiles
compatibles y elección del mejor candidato. Usa vectores numpy sintéticos
directamente (sin pasar por librosa/extraer_embedding), así que corre
rápido y no depende del motor de audio.
"""
import numpy as np
import pytest

import config
import matching


def _vec(valor: float, dim: int = None) -> np.ndarray:
    dim = dim or config.EMBEDDING_DIM
    return np.full(dim, valor, dtype=np.float64)


def test_filtrar_perfiles_compatibles_excluye_version_vieja():
    perfiles_todos = [
        ("Ana", _vec(0.1), config.EMBEDDING_VERSION),
        ("Luis", _vec(0.2, dim=60), config.EMBEDDING_VERSION - 1),  # embedding viejo, 60 dim
    ]
    compatibles, desactualizados = matching.filtrar_perfiles_compatibles(perfiles_todos)

    assert [n for n, _ in compatibles] == ["Ana"]
    assert desactualizados == ["Luis"]


def test_filtrar_perfiles_compatibles_excluye_dimension_distinta_aunque_version_coincida():
    """Caso más sutil: version_embedding correcta en la BD, pero el
    vector guardado no tiene la dimensión actual (p. ej. alguien cambió
    N_MFCC sin subir EMBEDDING_VERSION). Debe seguir excluyéndose."""
    perfiles_todos = [
        ("Marta", _vec(0.3, dim=config.EMBEDDING_DIM - 10), config.EMBEDDING_VERSION),
    ]
    compatibles, desactualizados = matching.filtrar_perfiles_compatibles(perfiles_todos)

    assert compatibles == []
    assert desactualizados == ["Marta"]


def test_identificar_mejor_candidato_sin_perfiles():
    resultado = matching.identificar_mejor_candidato(_vec(0.5), [])
    assert resultado.nombre is None
    assert resultado.identificado is False


def test_identificar_mejor_candidato_reconoce_coincidencia_exacta():
    embedding_nuevo = _vec(0.7)
    perfiles = [("Ana", _vec(0.7)), ("Luis", _vec(-0.7))]

    resultado = matching.identificar_mejor_candidato(embedding_nuevo, perfiles)

    assert resultado.nombre == "Ana"
    assert resultado.similitud == pytest.approx(1.0)
    assert resultado.identificado is True


def test_identificar_mejor_candidato_acepta_similitud_alta_aunque_distancia_sea_grande():
    """Misma dirección, distinta magnitud: similitud 100% pero distancia
    euclidiana grande. Solo decide la similitud, así que se identifica."""
    embedding_nuevo = _vec(1.0)
    perfiles = [("Emanuel", _vec(2.0))]

    resultado = matching.identificar_mejor_candidato(embedding_nuevo, perfiles)

    assert resultado.distancia > 0.30
    assert resultado.nombre == "Emanuel"
    assert resultado.identificado is True


def _con_similitud(coseno: float) -> np.ndarray:
    """Vector unitario cuya similitud coseno con e0 es exactamente `coseno`."""
    v = np.zeros(config.EMBEDDING_DIM)
    v[0], v[1] = coseno, np.sqrt(1 - coseno ** 2)
    return v


@pytest.mark.parametrize("coseno,esperado", [
    (0.90, False),    # el umbral anterior: aceptaba a personas distintas
    (0.9949, False),  # mejor caso entre personas distintas en la calibración sintética
    (0.996, True),
    (0.999, True),    # peor caso de la misma persona en la calibración sintética
])
def test_umbral_separa_misma_persona_de_personas_distintas(coseno, esperado):
    e0 = np.zeros(config.EMBEDDING_DIM)
    e0[0] = 1.0
    resultado = matching.identificar_mejor_candidato(e0, [("Ana", _con_similitud(coseno))])
    assert resultado.identificado is esperado


def test_identificar_mejor_candidato_rechaza_bajo_umbral():
    embedding_nuevo = _vec(1.0)
    # Vector deliberadamente distinto -> similitud coseno baja
    perfiles = [("Desconocido", np.concatenate([
        _vec(1.0, dim=config.EMBEDDING_DIM // 2),
        _vec(-1.0, dim=config.EMBEDDING_DIM - config.EMBEDDING_DIM // 2),
    ]))]

    resultado = matching.identificar_mejor_candidato(embedding_nuevo, perfiles)

    assert resultado.identificado is False
