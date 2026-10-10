"""
recalcular_embeddings.py
------------------------
Recalcula los embeddings de todas las personas registradas con el motor
actual (config.MOTOR_EMBEDDING), a partir de sus audios cifrados de
data/audio. Así nadie tiene que volver a grabarse al cambiar de motor:

    python recalcular_embeddings.py

  - Solo toca las muestras desactualizadas (otra versión o dimensión).
  - Cada audio se descifra en memoria; no se escribe ninguna copia.
  - Una muestra cuyo audio ya no está en disco se deja como está y se
    avisa (esa persona tendrá que re-entrenarse).
"""

import sys

import audio_processor as ap
import config
import database as db
import security


def recalcular() -> dict:
    """Devuelve un resumen: recalculadas, al_dia, sin_audio (nombres de personas)."""
    resumen = {"recalculadas": 0, "al_dia": 0, "sin_audio": []}
    proyeccion = {"nombre": 1, "muestras.archivo_audio": 1, "muestras.version_embedding": 1, "muestras.dim": 1}
    for doc in db._speakers().find({}, proyeccion):
        for muestra in doc.get("muestras", []):
            if muestra.get("version_embedding") == config.EMBEDDING_VERSION and muestra.get("dim") == config.EMBEDDING_DIM:
                resumen["al_dia"] += 1
                continue
            archivo = muestra["archivo_audio"]
            try:
                embedding = ap.extraer_embedding(ap.cargar_audio_desde_archivo(db.ruta_audio_segura(archivo)))
            except FileNotFoundError:
                resumen["sin_audio"].append(doc["nombre"])
                continue
            db._speakers().update_one(
                {"_id": doc["_id"], "muestras.archivo_audio": archivo},
                {"$set": {"muestras.$.embedding": db._cifrar_embedding(embedding),
                          "muestras.$.version_embedding": int(config.EMBEDDING_VERSION),
                          "muestras.$.dim": int(embedding.shape[0])}},
            )
            resumen["recalculadas"] += 1
    return resumen


def main() -> None:
    security.configurar_logs()
    try:
        security.verificar_clave_cifrado()
        ap.verificar_motor()
        db.init_db()
    except (db.BaseDatosError, RuntimeError) as e:
        print(e)
        sys.exit(1)
    if not security.pedir_acceso():     # descifra datos biométricos: exige el PIN
        sys.exit(1)
    print(f"Recalculando con el motor '{config.MOTOR_EMBEDDING}' (versión {config.EMBEDDING_VERSION})...")
    try:
        resumen = recalcular()
    except db.BaseDatosError as e:
        print(e)
        sys.exit(1)
    print(f"Muestras recalculadas: {resumen['recalculadas']} | ya al día: {resumen['al_dia']}")
    if resumen["sin_audio"]:
        print(f"Sin audio en data/audio (re-entrénalas): {', '.join(sorted(set(resumen['sin_audio'])))}")


if __name__ == "__main__":
    main()
