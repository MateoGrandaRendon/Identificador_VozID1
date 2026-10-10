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


@pytest.mark.parametrize("motor,coseno,esperado", [
    # ECAPA, valores medidos con voces reales (2026-10-10)
    ("ecapa", 0.094, False),   # otra persona: el caso más parecido medido
    ("ecapa", 0.39, False),    # justo bajo el umbral
    ("ecapa", 0.63, True),     # la misma persona, grabada otro día (peor caso medido)
    ("ecapa", 0.77, True),     # la misma persona, misma sesión
    # Clásico (solo calibrado con voces sintéticas)
    ("clasico", 0.90, False),
    ("clasico", 0.9949, False),
    ("clasico", 0.996, True),
])
def test_umbral_separa_misma_persona_de_personas_distintas(monkeypatch, motor, coseno, esperado):
    monkeypatch.setattr(config, "UMBRAL_SIMILITUD", config._UMBRALES[motor])
    e0 = np.zeros(config.EMBEDDING_DIM)
    e0[0] = 1.0
    resultado = matching.identificar_mejor_candidato(e0, [("Ana", _con_similitud(coseno))])
    assert resultado.identificado is esperado


def _e0():
    e0 = np.zeros(config.EMBEDDING_DIM)
    e0[0] = 1.0
    return e0


def _perfil(nombre, cosenos):
    return [(nombre, _con_similitud(c)) for c in cosenos]


def test_promedio_de_4_no_se_deja_enganar_por_una_muestra_casual():
    """Ana tiene UNA muestra casi idéntica por casualidad y el resto lejos;
    Luis es consistentemente parecido. Con el máximo ganaba Ana; con el
    promedio de las 4 más parecidas gana Luis."""
    perfiles = _perfil("Ana", [0.9999, 0.95, 0.95, 0.95, 0.95]) + _perfil("Luis", [0.998] * 4)
    resultado = matching.identificar_mejor_candidato(_e0(), perfiles)
    assert resultado.nombre == "Luis"
    assert resultado.similitud == pytest.approx(0.998)


def test_promedio_usa_solo_las_4_mejores_muestras():
    perfiles = _perfil("Ana", [0.999, 0.999, 0.999, 0.999, 0.50, 0.40])
    puntuacion = matching.puntuar_personas(_e0(), perfiles)[0]
    assert puntuacion.similitud == pytest.approx(0.999)


def test_persona_con_menos_de_4_muestras_compatibles_promedia_las_que_tiene():
    puntuacion = matching.puntuar_personas(_e0(), _perfil("Ana", [0.998, 0.996]))[0]
    assert puntuacion.similitud == pytest.approx(0.997)


def test_margen_insuficiente_frente_al_segundo_no_identifica():
    """Dos personas casi igual de parecidas: es ambiguo, no se asigna a nadie."""
    perfiles = _perfil("Ana", [0.9980] * 4) + _perfil("Luis", [0.9979] * 4)
    resultado = matching.identificar_mejor_candidato(_e0(), perfiles)
    assert resultado.nombre == "Ana" and resultado.segundo == "Luis"
    assert resultado.margen < config.MARGEN_DISTANCIA
    assert resultado.identificado is False


def test_margen_claro_frente_al_segundo_identifica():
    perfiles = _perfil("Ana", [0.999] * 4) + _perfil("Luis", [0.990] * 4)
    resultado = matching.identificar_mejor_candidato(_e0(), perfiles)
    # distancia ∝ sqrt(1 - coseno): sqrt(0.010 / 0.001) ≈ 3.16 veces más lejos
    assert resultado.margen == pytest.approx(np.sqrt(10), rel=1e-3)
    assert resultado.identificado is True


def test_coincidencia_exacta_tiene_margen_infinito():
    perfiles = _perfil("Ana", [1.0] * 4) + _perfil("Luis", [0.99] * 4)
    resultado = matching.identificar_mejor_candidato(_e0(), perfiles)
    assert resultado.margen == float("inf") and resultado.identificado is True


def test_una_sola_persona_no_tiene_margen():
    resultado = matching.identificar_mejor_candidato(_e0(), _perfil("Ana", [0.999] * 4))
    assert resultado.segundo is None and resultado.margen is None and resultado.identificado is True


def test_identificar_mejor_candidato_rechaza_bajo_umbral():
    embedding_nuevo = _vec(1.0)
    # Vector deliberadamente distinto -> similitud coseno baja
    perfiles = [("Desconocido", np.concatenate([
        _vec(1.0, dim=config.EMBEDDING_DIM // 2),
        _vec(-1.0, dim=config.EMBEDDING_DIM - config.EMBEDDING_DIM // 2),
    ]))]

    resultado = matching.identificar_mejor_candidato(embedding_nuevo, perfiles)

    assert resultado.identificado is False
