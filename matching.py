"""
matching.py
-----------
Lógica de comparación/decisión de identificación de voz, separada de la
capa de I/O (CLI en main.py).

Por qué existe como módulo aparte:
 - Se puede probar con pytest sin necesitar micrófono, teclado ni una
   base de datos real (ver test/test_matching.py).
 - Cuando llegue la Fase 2 (API FastAPI/Flask) o una GUI, ambas
   reutilizan EXACTAMENTE esta misma lógica de decisión en vez de
   reimplementarla — una sola fuente de verdad para "¿esta voz es esta
   persona?".

Regla de decisión (ver config.MUESTRAS_COMPARADAS y config.MARGEN_DISTANCIA):
 1. Para cada persona se promedian las MUESTRAS_COMPARADAS (4) similitudes
    más altas entre la voz nueva y sus muestras. Con el máximo de UNA sola
    muestra, bastaba con que una de las hasta 25 grabaciones se pareciera
    por casualidad para aceptar a alguien.
 2. Se acepta a la mejor persona si su promedio alcanza UMBRAL_SIMILITUD
    Y la segunda persona está al menos MARGEN_DISTANCIA veces más lejos
    (distancia euclidiana media de esas mismas muestras). Así una voz no
    registrada que "se parece un poco a todos" no se asigna a nadie.
"""

from collections import defaultdict
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
    segundo: str | None = None   # segunda persona más parecida (None si solo hay una)
    margen: float | None = None  # distancia de la 2ª / distancia de la 1ª (None si solo hay una)


@dataclass
class PuntuacionPersona:
    nombre: str
    similitud: float   # promedio de las MUESTRAS_COMPARADAS similitudes más altas
    distancia: float   # distancia euclidiana media de esas mismas muestras


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


def puntuar_personas(embedding_nuevo: np.ndarray, perfiles_compatibles, k: int = None) -> list[PuntuacionPersona]:
    """
    Puntúa a cada persona con el promedio de sus k muestras más parecidas
    (todas si tiene menos de k compatibles). Devuelve la lista ordenada de
    mayor a menor similitud. Nunca lanza excepción por dimensiones distintas:
    similitud_coseno y distancia_euclidiana ya son defensivas ante eso.
    """
    k = k or config.MUESTRAS_COMPARADAS
    por_persona = defaultdict(list)
    for nombre, embedding in perfiles_compatibles:
        por_persona[nombre].append((ap.similitud_coseno(embedding_nuevo, embedding),
                                    ap.distancia_euclidiana(embedding_nuevo, embedding)))
    puntuaciones = []
    for nombre, pares in por_persona.items():
        mejores = sorted(pares, key=lambda par: par[0], reverse=True)[:k]
        puntuaciones.append(PuntuacionPersona(
            nombre,
            float(np.mean([s for s, _ in mejores])),
            float(np.mean([d for _, d in mejores])),
        ))
    return sorted(puntuaciones, key=lambda p: p.similitud, reverse=True)


def margen_distancia(primera: PuntuacionPersona, segunda: PuntuacionPersona) -> float:
    """Cuántas veces más lejos está la segunda persona que la primera (inf si la primera coincide exacta)."""
    if primera.distancia <= 0:
        return float("inf") if segunda.distancia > 0 else 1.0
    return segunda.distancia / primera.distancia


def decidir(puntuaciones: list[PuntuacionPersona], umbral: float = None, margen_min: float = None) -> ResultadoIdentificacion:
    """Aplica la regla umbral + margen a unas puntuaciones ya ordenadas (ver puntuar_personas)."""
    umbral = config.UMBRAL_SIMILITUD if umbral is None else umbral
    margen_min = config.MARGEN_DISTANCIA if margen_min is None else margen_min
    if not puntuaciones:
        return ResultadoIdentificacion(None, -1.0, float("inf"), False)
    mejor = puntuaciones[0]
    segundo = puntuaciones[1] if len(puntuaciones) > 1 else None
    margen = margen_distancia(mejor, segundo) if segundo else None
    identificado = mejor.similitud >= umbral and (margen is None or margen >= margen_min)
    return ResultadoIdentificacion(mejor.nombre, mejor.similitud, mejor.distancia, identificado,
                                   segundo.nombre if segundo else None, margen)


def identificar_mejor_candidato(embedding_nuevo: np.ndarray, perfiles_compatibles) -> ResultadoIdentificacion:
    """
    Compara embedding_nuevo contra los perfiles ya filtrados como
    compatibles con la regla del docstring del módulo. Si
    perfiles_compatibles está vacío, devuelve identificado=False con
    nombre=None.
    """
    return decidir(puntuar_personas(embedding_nuevo, perfiles_compatibles))
