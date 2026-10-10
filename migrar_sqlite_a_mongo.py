"""
migrar_sqlite_a_mongo.py
------------------------
Migra los datos de la versión anterior (SQLite en data/voice_id.db +
embeddings .npy.enc en disco) a MongoDB. Se ejecuta UNA vez:

    python migrar_sqlite_a_mongo.py              # migra y conserva el .db
    python migrar_sqlite_a_mongo.py --borrar-sqlite

  - Usa la misma clave de encriptación del .env (no re-cifra nada a mano).
  - Renombra cada audio a un identificador aleatorio (sin el nombre de la
    persona) y guarda el embedding cifrado dentro del documento de MongoDB.
  - Si una persona tiene más de 25 muestras se conservan las 25 más
    recientes; si tiene menos de 4 no se puede migrar (no cumple el mínimo)
    y se indica que hay que registrarla de nuevo.
  - Con --borrar-sqlite elimina el .db y los .npy.enc antiguos al terminar:
    dejar copias de la base anterior es una exposición innecesaria de datos.
"""

import datetime
import sqlite3
import sys
import uuid
from pathlib import Path

from bson import Binary
from pymongo.errors import DuplicateKeyError

import config
import database as db
import security


def _ruta_legada(ruta: str) -> Path | None:
    """Solo acepta archivos dentro de data/audio (la BD antigua guardaba rutas completas)."""
    p = Path(ruta).resolve()
    return p if p.parent == config.AUDIO_DIR.resolve() and p.is_file() else None


def _fecha(valor) -> datetime.datetime:
    try:
        f = datetime.datetime.fromisoformat(str(valor))
    except ValueError:
        return datetime.datetime.now(datetime.timezone.utc)
    return f.replace(tzinfo=datetime.timezone.utc) if f.tzinfo is None else f


def migrar(borrar_sqlite: bool) -> None:
    if not config.SQLITE_LEGADO_PATH.exists():
        print("No hay base de datos SQLite antigua que migrar.")
        return
    db.init_db()
    con = sqlite3.connect(config.SQLITE_LEGADO_PATH)
    con.row_factory = sqlite3.Row
    columnas = {fila[1] for fila in con.execute("PRAGMA table_info(voice_samples)")}
    col_version = "version_embedding" if "version_embedding" in columnas else "0 AS version_embedding"

    migradas, omitidas, embeddings_viejos = 0, [], []
    for sp in con.execute("SELECT id, nombre, fecha_registro FROM speakers"):
        # Mismas reglas de nombre que la GUI y el CLI: así ningún nombre migrado puede
        # chocar con el nombre reservado de las pruebas (main.NOMBRE_PRUEBA, que empieza
        # por "_" y se borra en cada autoprueba) ni quedar imposible de gestionar.
        try:
            nombre = security.validar_nombre_persona(str(sp["nombre"])[: config.NOMBRE_MAX_LARGO])
        except ValueError:
            omitidas.append(f"registro #{sp['id']} (nombre no válido: regístralo de nuevo)")
            continue
        filas = con.execute(
            f"SELECT ruta_audio, ruta_embedding, fecha_creacion, {col_version} "
            "FROM voice_samples WHERE speaker_id = ? ORDER BY fecha_creacion DESC",
            (sp["id"],),
        ).fetchall()[: config.MAX_MUESTRAS_POR_PERSONA]

        muestras, renombres, embeddings_persona = [], [], []
        for f in filas:
            audio, emb = _ruta_legada(f["ruta_audio"]), _ruta_legada(f["ruta_embedding"])
            if audio is None or emb is None:
                continue
            try:
                embedding = db._descifrar_embedding(emb.read_bytes())
            except Exception:
                continue
            nuevo = f"{uuid.uuid4().hex}.wav.enc"
            renombres.append((audio, config.AUDIO_DIR / nuevo))
            embeddings_persona.append(emb)
            muestras.append({
                "archivo_audio": nuevo,
                "embedding": Binary(emb.read_bytes()),  # ya está cifrado con la misma clave
                "version_embedding": int(f["version_embedding"] or 0),
                "dim": int(embedding.shape[0]),
                "fecha_creacion": _fecha(f["fecha_creacion"]),
            })

        if len(muestras) < config.MIN_MUESTRAS_POR_PERSONA:
            omitidas.append(nombre)
            continue
        try:
            db._speakers().insert_one({
                "nombre": nombre,
                "nombre_clave": db._clave(nombre),
                "fecha_registro": _fecha(sp["fecha_registro"]),
                "muestras": list(reversed(muestras)),
            })
        except DuplicateKeyError:
            omitidas.append(f"{nombre} (ya existía en MongoDB)")
            continue
        for viejo, nuevo in renombres:
            viejo.rename(nuevo)
        # Solo los embeddings de personas migradas: los de las omitidas se conservan.
        embeddings_viejos.extend(embeddings_persona)
        migradas += 1

    con.close()
    print(f"✅ Personas migradas a MongoDB: {migradas}")
    if omitidas:
        print(f"⚠️  No migradas (menos de {config.MIN_MUESTRAS_POR_PERSONA} muestras legibles o duplicadas):")
        for n in omitidas:
            print(f"   - {n}")

    if borrar_sqlite:
        for emb in embeddings_viejos:
            emb.unlink(missing_ok=True)
        config.SQLITE_LEGADO_PATH.unlink()
        print("🗑️  Base SQLite antigua y embeddings .npy.enc eliminados.")
    else:
        print(f"ℹ️  Cuando verifiques la migración, borra {config.SQLITE_LEGADO_PATH.name} "
              "(o vuelve a ejecutar con --borrar-sqlite): es una copia innecesaria de datos personales.")


if __name__ == "__main__":
    security.configurar_logs()
    try:
        migrar("--borrar-sqlite" in sys.argv[1:])
    except db.BaseDatosError as e:
        print(f"❌ {e}")
        sys.exit(1)
