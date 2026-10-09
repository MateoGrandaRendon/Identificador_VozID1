"""
registro.py
-----------
Persistencia de un perfil de voz completo, compartida por el CLI (main.py)
y la interfaz gráfica (frontend/bridge.py), para que ambos apliquen
exactamente las mismas reglas:

  - "registro":   persona nueva, entre MIN y MAX muestras.
  - "reentrenar": reemplaza todas las muestras, entre MIN y MAX.
  - "ampliar":    añade muestras sin pasar del MAX en total.

Primero se escriben los audios cifrados y se extraen los embeddings; luego
se hace UNA escritura atómica en MongoDB. Si algo falla, se borran los
audios recién creados para no dejar archivos huérfanos.
"""

from pathlib import Path

import audio_processor as ap
import config
import database as db

MODOS = ("registro", "reentrenar", "ampliar")


def limites_nuevas_muestras(modo: str, muestras_actuales: int = 0) -> tuple[int, int]:
    """(mínimo, máximo) de muestras NUEVAS a grabar en el modo dado."""
    if modo == "ampliar":
        return 1, max(0, config.MAX_MUESTRAS_POR_PERSONA - muestras_actuales)
    return config.MIN_MUESTRAS_POR_PERSONA, config.MAX_MUESTRAS_POR_PERSONA


def guardar_perfil(nombre: str, audios, modo: str) -> int:
    """
    Guarda los audios como muestras de `nombre` según `modo`. Devuelve
    cuántas muestras se guardaron. Lanza db.BaseDatosError (mensaje apto
    para el usuario) si no se cumple alguna regla.
    """
    if modo not in MODOS:
        raise db.BaseDatosError("Modo de registro inválido.")
    rutas = []
    try:
        muestras = []
        for audio in audios:
            ruta = ap.guardar_wav(audio)
            rutas.append(ruta)
            muestras.append((ruta, ap.extraer_embedding(audio)))

        if modo == "registro":
            if db.registrar_speaker(nombre, muestras) == "duplicado":
                raise db.BaseDatosError(f"Ya existe una persona llamada '{nombre}'.")
        elif modo == "reentrenar":
            antiguos = db.reemplazar_muestras(nombre, muestras)
            if antiguos is None:
                raise db.BaseDatosError(f"No se encontró a '{nombre}'.")
            db.borrar_archivos_audio(antiguos)
        else:
            resultado = db.agregar_muestras(nombre, muestras)
            if resultado == "no_existe":
                raise db.BaseDatosError(f"No se encontró a '{nombre}'.")
            if resultado == "limite":
                raise db.LimiteMuestrasError(
                    f"'{nombre}' no puede superar {config.MAX_MUESTRAS_POR_PERSONA} muestras de voz."
                )
        return len(muestras)
    except Exception:
        db.borrar_archivos_audio(Path(r).name for r in rutas)
        raise
