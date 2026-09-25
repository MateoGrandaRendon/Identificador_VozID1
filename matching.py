"""
matching.py
-----------
Lógica de comparación/decisión de identificación de voz, separada de la
capa de I/O (CLI en main.py).

Por qué existe como módulo aparte:
 - Se puede probar con pytest sin necesitar micrófono, teclado ni una
   base de datos real (ver tests/test_matching.py).
 - Cuando llegue la Fase 2 (API FastAPI/Flask) o una GUI, ambas
   reutilizan EXACTAMENTE esta misma lógica de decisión en vez de
   reimplementarla — una sola fuente de verdad para "¿esta voz es esta
   persona?".
"""

from dataclasses import dataclass

import numpy as np

import config
import audio_processor as ap


@dataclass
class ResultadoIdentificacion:
    nombre: str | None
    similitud: float
    distancia: float
    identificado: bool


def filtrar_perfiles_compatibles(perfiles_todos):
    """
    Separa los perfiles cargados desde la BD (nombre, embedding, version)
    en:
      - compatibles: [(nombre, embedding), ...] generados con la
        versión Y dimensión VIGENTES del pipeline (config.EMBEDDING_VERSION
        / config.EMBEDDING_DIM) — los únicos seguros para comparar.
      - desactualizados: nombres (sin duplicados, ordenados) de personas
        con AL MENOS una muestra que no cumple lo anterior.

    Se valida versión Y dimensión (no solo la versión) a propósito: así
    queda protegido incluso el caso en que alguien cambie un parámetro
    del pipeline (p. ej. N_MFCC) sin recordar subir EMBEDDING_VERSION en
    config.py. Es exactamente lo que evita el error histórico
    "shapes (156,) and (60,) not aligned".
    """
    compatibles = [
        (nombre, emb) for nombre, emb, version in perfiles_todos
        if version == config.EMBEDDING_VERSION and emb.shape[0] == config.EMBEDDING_DIM
    ]
    desactualizados = sorted({
        nombre for nombre, emb, version in perfiles_todos
        if version != config.EMBEDDING_VERSION or emb.shape[0] != config.EMBEDDING_DIM
    })
    return compatibles, desactualizados


def identificar_mejor_candidato(embedding_nuevo: np.ndarray, perfiles_compatibles) -> ResultadoIdentificacion:
    """
    Compara embedding_nuevo contra cada perfil ya filtrado como
    compatible, con el motor ESTRICTO (similitud coseno Y distancia
    euclidiana deben cumplir su umbral a la vez — ver config.py). Se
    elige como candidato el de mayor similitud coseno; su distancia
    euclidiana se evalúa después como segundo criterio obligatorio, no
    como desempate.

    Nunca lanza excepción por dimensiones distintas: similitud_coseno y
    distancia_euclidiana (audio_processor.py) ya son defensivas ante eso.
    Si perfiles_compatibles está vacío, devuelve identificado=False con
    nombre=None.
    """
    mejor_nombre, mejor_similitud, mejor_distancia = None, -1.0, float("inf")

    for nombre, embedding in perfiles_compatibles:
        similitud = ap.similitud_coseno(embedding_nuevo, embedding)
        distancia = ap.distancia_euclidiana(embedding_nuevo, embedding)
        if similitud > mejor_similitud:
            mejor_similitud, mejor_distancia, mejor_nombre = similitud, distancia, nombre

    identificado = (
        mejor_nombre is not None
        and mejor_similitud >= config.UMBRAL_SIMILITUD
        and mejor_distancia <= config.UMBRAL_DISTANCIA_EUCLIDIANA
    )
    return ResultadoIdentificacion(mejor_nombre, mejor_similitud, mejor_distancia, identificado)
