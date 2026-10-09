"""
database.py
-----------
Capa de acceso a datos sobre MongoDB (base de datos NO relacional, pymongo).

Modelo de datos — colección `speakers`, un documento por persona con sus
muestras de voz EMBEBIDAS (relación "uno a pocos", el patrón idiomático en
MongoDB):

    {
      _id:            ObjectId,
      nombre:         "Ana María",          # tal como se muestra
      nombre_clave:   "ana maría",          # normalizado; índice ÚNICO
      fecha_registro: ISODate,
      muestras: [                           # entre 4 y 25 elementos
        {
          archivo_audio:     "<uuid>.wav.enc",   # solo el NOMBRE, nunca una ruta
          embedding:         BinData,            # vector numpy ENCRIPTADO (Fernet)
          version_embedding: 2,
          dim:               156,
          fecha_creacion:    ISODate,
        }, ...
      ]
    }

Por qué embebido y no en una colección aparte:
 - Toda escritura sobre UN documento es atómica en MongoDB: registrar,
   re-entrenar o ampliar las muestras de una persona nunca deja estados a
   medias, sin necesitar transacciones (que exigen un replica set).
 - El límite de 4 a 25 muestras se hace cumplir en la propia base de datos
   con un validador $jsonSchema (minItems / maxItems), además de en el
   código: ni un cliente con errores puede saltárselo.
 - 25 embeddings encriptados ocupan ~50 KB: muy lejos del límite de 16 MB
   por documento.

Seguridad (inyección NoSQL): todos los filtros se construyen aquí con
valores que se verifican como `str`. Nunca se pasa a pymongo un
dict recibido desde fuera, así que un valor como {"$ne": ""} no puede
convertirse en un operador de consulta.

El audio (más pesado) sigue en disco, encriptado, en data/audio/; la base
de datos solo guarda el nombre del archivo, que se valida contra un patrón
estricto antes de tocar el disco (evita recorrido de directorios).
"""

import datetime
import io
import logging
import re
import threading
from pathlib import Path

import numpy as np
from bson import Binary
from cryptography.fernet import InvalidToken
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import (
    CollectionInvalid,
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
)
from pymongo.uri_parser import parse_uri

import config
import security

log = logging.getLogger("voiceid.db")

COLECCION_SPEAKERS = "speakers"
COLECCION_SEGURIDAD = "estado_seguridad"

# Nombre de archivo de audio válido: generado por audio_processor.guardar_wav
# (uuid4 en hexadecimal). Cualquier otra cosa (p. ej. "../../x") se rechaza.
_PATRON_ARCHIVO_AUDIO = re.compile(r"^[0-9a-f]{32}\.wav\.enc$")
_HOSTS_LOCALES = {"localhost", "127.0.0.1", "::1"}


class BaseDatosError(Exception):
    """Error de la capa de datos con un mensaje apto para mostrar al usuario."""


class LimiteMuestrasError(BaseDatosError):
    """Se intentó guardar menos del mínimo o más del máximo de muestras por persona."""


# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------
_cliente = None
_lock_cliente = threading.Lock()


def _validar_uri(uri: str) -> None:
    """
    Rechaza conexiones inseguras a servidores remotos: si algún host no es
    local, la conexión debe ir cifrada (TLS) y con certificados verificados.
    mongodb+srv:// activa TLS por defecto en pymongo.
    """
    if uri.startswith("mongodb+srv://"):
        partes = parse_uri(uri.replace("mongodb+srv://", "mongodb://", 1), validate=False)
        opciones = partes["options"]
        remoto, tls = True, opciones.get("tls", opciones.get("ssl", True))
    else:
        partes = parse_uri(uri)
        opciones = partes["options"]
        remoto = any(host not in _HOSTS_LOCALES for host, _ in partes["nodelist"])
        tls = opciones.get("tls", opciones.get("ssl", False))

    if remoto and not tls:
        raise BaseDatosError(
            "La conexión a un MongoDB remoto debe usar TLS. Usa una URI mongodb+srv:// "
            "o añade 'tls=true' a VOICE_ID_MONGODB_URI en el .env."
        )
    if any(opciones.get(k) for k in ("tlsInsecure", "tlsAllowInvalidCertificates", "tlsAllowInvalidHostnames")):
        raise BaseDatosError("La URI de MongoDB desactiva la verificación de certificados TLS: no está permitido.")


def _db():
    """Devuelve la base de datos, creando el cliente (con pool) la primera vez."""
    global _cliente
    if _cliente is None:
        with _lock_cliente:
            if _cliente is None:
                _validar_uri(config.MONGODB_URI)
                _cliente = MongoClient(
                    config.MONGODB_URI,
                    appname="VoiceID",
                    tz_aware=True,
                    serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS,
                    connectTimeoutMS=config.MONGODB_TIMEOUT_MS,
                    socketTimeoutMS=config.MONGODB_TIMEOUT_MS * 4,
                    maxPoolSize=10,
                )
    return _cliente[config.MONGODB_DB]


def usar_cliente(cliente) -> None:
    """Inyecta un cliente ya creado (p. ej. mongomock en las pruebas)."""
    global _cliente
    _cliente = cliente


def _speakers():
    return _db()[COLECCION_SPEAKERS]


_ESQUEMA_SPEAKERS = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["nombre", "nombre_clave", "fecha_registro", "muestras"],
        "properties": {
            "nombre": {"bsonType": "string", "minLength": 1, "maxLength": config.NOMBRE_MAX_LARGO},
            "nombre_clave": {"bsonType": "string", "minLength": 1, "maxLength": config.NOMBRE_MAX_LARGO},
            "fecha_registro": {"bsonType": "date"},
            "muestras": {
                "bsonType": "array",
                "minItems": config.MIN_MUESTRAS_POR_PERSONA,
                "maxItems": config.MAX_MUESTRAS_POR_PERSONA,
                "items": {
                    "bsonType": "object",
                    "required": ["archivo_audio", "embedding", "version_embedding", "dim", "fecha_creacion"],
                    "properties": {
                        "archivo_audio": {"bsonType": "string", "pattern": _PATRON_ARCHIVO_AUDIO.pattern},
                        "embedding": {"bsonType": "binData"},
                        "version_embedding": {"bsonType": "int"},
                        "dim": {"bsonType": "int"},
                        "fecha_creacion": {"bsonType": "date"},
                    },
                },
            },
        },
    }
}


def init_db() -> None:
    """
    Verifica la conexión, crea la colección con su validador $jsonSchema y
    los índices. Es idempotente. Lanza BaseDatosError si MongoDB no responde.
    """
    try:
        db = _db()
        db.command("ping")
    except PyMongoError as e:
        log.error("No se pudo conectar a MongoDB: %s", type(e).__name__)
        raise BaseDatosError(
            "No se pudo conectar a MongoDB. Verifica que el servidor esté en marcha "
            "y que VOICE_ID_MONGODB_URI en el .env sea correcta."
        ) from None

    try:
        db.create_collection(COLECCION_SPEAKERS, validator=_ESQUEMA_SPEAKERS, validationLevel="strict")
    except CollectionInvalid:
        # Ya existía: se actualiza el validador por si cambiaron los límites.
        try:
            db.command({"collMod": COLECCION_SPEAKERS, "validator": _ESQUEMA_SPEAKERS, "validationLevel": "strict"})
        except (PyMongoError, NotImplementedError) as e:
            log.warning("No se pudo actualizar el validador de esquema: %s", type(e).__name__)
    except (OperationFailure, NotImplementedError) as e:
        # Usuario de MongoDB sin permiso dbAdmin: el código sigue validando los límites.
        log.warning("No se pudo crear la colección con validador: %s", type(e).__name__)

    _speakers().create_index("nombre_clave", unique=True, name="nombre_unico")


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _clave(nombre: str) -> str:
    """Clave de búsqueda/unicidad: 'Ana' y 'ana' se consideran la misma persona."""
    if not isinstance(nombre, str):
        raise BaseDatosError("Nombre inválido.")
    clave = nombre.strip().casefold()
    if not clave or len(clave) > config.NOMBRE_MAX_LARGO:
        raise BaseDatosError("Nombre inválido.")
    return clave


def _ahora() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def ruta_audio_segura(archivo_audio: str) -> Path:
    """
    Convierte el nombre de archivo guardado en la BD en una ruta dentro de
    config.AUDIO_DIR, verificando que no pueda escapar de esa carpeta.
    """
    if not isinstance(archivo_audio, str) or not _PATRON_ARCHIVO_AUDIO.fullmatch(archivo_audio):
        raise BaseDatosError("Nombre de archivo de audio inválido.")
    base = config.AUDIO_DIR.resolve()
    ruta = (base / archivo_audio).resolve()
    if ruta.parent != base:
        raise BaseDatosError("Nombre de archivo de audio inválido.")
    return ruta


def borrar_archivos_audio(archivos) -> int:
    """Borra del disco los audios indicados (solo dentro de AUDIO_DIR). Devuelve cuántos se borraron."""
    borrados = 0
    for archivo in archivos:
        try:
            ruta_audio_segura(archivo).unlink(missing_ok=True)
            borrados += 1
        except (OSError, BaseDatosError):
            log.warning("No se pudo borrar un archivo de audio")
    return borrados


def _cifrar_embedding(embedding: np.ndarray) -> Binary:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(embedding, dtype=np.float64), allow_pickle=False)
    return Binary(security.encriptar_bytes(buffer.getvalue()))


def _descifrar_embedding(datos: bytes) -> np.ndarray:
    # allow_pickle=False: un .npy manipulado nunca puede ejecutar código al cargarse.
    return np.load(io.BytesIO(security.desencriptar_bytes(bytes(datos))), allow_pickle=False)


def _doc_muestra(ruta_audio, embedding: np.ndarray) -> dict:
    archivo = Path(ruta_audio).name
    ruta_audio_segura(archivo)  # valida el nombre
    return {
        "archivo_audio": archivo,
        "embedding": _cifrar_embedding(embedding),
        "version_embedding": int(config.EMBEDDING_VERSION),
        "dim": int(np.asarray(embedding).shape[0]),
        "fecha_creacion": _ahora(),
    }


def _validar_cantidad(n: int, minimo: int, maximo: int) -> None:
    if n < minimo or n > maximo:
        raise LimiteMuestrasError(
            f"Cada persona debe tener entre {config.MIN_MUESTRAS_POR_PERSONA} y "
            f"{config.MAX_MUESTRAS_POR_PERSONA} muestras de voz (se intentó guardar {n})."
        )


# ---------------------------------------------------------------------------
# Operaciones de personas
# ---------------------------------------------------------------------------
def obtener_speaker(nombre: str) -> dict | None:
    """Devuelve el resumen de una persona (sin embeddings) o None si no existe."""
    doc = _speakers().find_one(
        {"nombre_clave": _clave(nombre)},
        {"nombre": 1, "fecha_registro": 1, "muestras.archivo_audio": 1, "muestras.fecha_creacion": 1},
    )
    if doc is None:
        return None
    return {
        "id": str(doc["_id"]),
        "nombre": doc["nombre"],
        "fecha_registro": doc["fecha_registro"],
        "muestras": doc.get("muestras", []),
    }


def registrar_speaker(nombre: str, muestras) -> str:
    """
    Crea una persona nueva con todas sus muestras en UNA sola escritura.
    `muestras` es una lista de (ruta_audio, embedding) con entre MIN y MAX
    elementos. Devuelve "ok" o "duplicado".
    """
    muestras = list(muestras)
    _validar_cantidad(len(muestras), config.MIN_MUESTRAS_POR_PERSONA, config.MAX_MUESTRAS_POR_PERSONA)
    try:
        _speakers().insert_one({
            "nombre": nombre.strip(),
            "nombre_clave": _clave(nombre),
            "fecha_registro": _ahora(),
            "muestras": [_doc_muestra(r, e) for r, e in muestras],
        })
    except DuplicateKeyError:
        return "duplicado"
    return "ok"


def reemplazar_muestras(nombre: str, muestras) -> list[str] | None:
    """
    Re-entrenamiento: sustituye atómicamente TODAS las muestras de una
    persona por las nuevas (entre MIN y MAX). Devuelve los nombres de los
    audios antiguos (para que quien llama los borre del disco) o None si la
    persona no existe.
    """
    muestras = list(muestras)
    _validar_cantidad(len(muestras), config.MIN_MUESTRAS_POR_PERSONA, config.MAX_MUESTRAS_POR_PERSONA)
    anterior = _speakers().find_one_and_update(
        {"nombre_clave": _clave(nombre)},
        {"$set": {"muestras": [_doc_muestra(r, e) for r, e in muestras]}},
        projection={"muestras.archivo_audio": 1},
        return_document=ReturnDocument.BEFORE,
    )
    if anterior is None:
        return None
    return [m["archivo_audio"] for m in anterior.get("muestras", [])]


def agregar_muestras(nombre: str, muestras) -> str:
    """
    Añade muestras a una persona existente sin superar el máximo.

    La condición del límite va DENTRO del filtro de la actualización
    ("el elemento en la posición MAX-k no existe" ⇔ "hay como mucho MAX-k
    muestras"), así que dos ventanas añadiendo a la vez no pueden pasar de
    25 entre ambas. Devuelve "ok", "no_existe" o "limite".
    """
    muestras = list(muestras)
    k = len(muestras)
    _validar_cantidad(k, 1, config.MAX_MUESTRAS_POR_PERSONA)
    clave = _clave(nombre)
    resultado = _speakers().update_one(
        {"nombre_clave": clave, f"muestras.{config.MAX_MUESTRAS_POR_PERSONA - k}": {"$exists": False}},
        {"$push": {"muestras": {"$each": [_doc_muestra(r, e) for r, e in muestras]}}},
    )
    if resultado.matched_count == 1:
        return "ok"
    return "limite" if _speakers().count_documents({"nombre_clave": clave}, limit=1) else "no_existe"


def listar_speakers() -> list[dict]:
    """
    Lista todas las personas con su diagnóstico de compatibilidad, SIN
    descifrar ningún embedding (version y dim se guardan en claro).
    """
    proyeccion = {"nombre": 1, "fecha_registro": 1, "muestras.version_embedding": 1, "muestras.dim": 1}
    resultado = []
    for doc in _speakers().find({}, proyeccion).sort("nombre_clave", 1):
        muestras = doc.get("muestras", [])
        compatibles = sum(
            1 for m in muestras
            if m.get("version_embedding") == config.EMBEDDING_VERSION and m.get("dim") == config.EMBEDDING_DIM
        )
        resultado.append({
            "id": str(doc["_id"]),
            "nombre": doc["nombre"],
            "fecha_registro": doc["fecha_registro"],
            "total_muestras": len(muestras),
            "muestras_compatibles": compatibles,
            "necesita_reentrenar": len(muestras) > 0 and compatibles == 0,
        })
    return resultado


def diagnostico_embeddings() -> list[dict]:
    """Compatibilidad de las muestras de cada persona con el motor vigente."""
    return [
        {k: p[k] for k in ("nombre", "total_muestras", "muestras_compatibles", "necesita_reentrenar")}
        for p in listar_speakers()
    ]


def obtener_embeddings_todos() -> list[tuple]:
    """
    Carga y DESCIFRA todos los embeddings. Devuelve una lista de tuplas
    (nombre_speaker, embedding_numpy, version_embedding). Una muestra que
    no se pueda descifrar (clave distinta o dato alterado) se omite.
    """
    resultados = []
    proyeccion = {"nombre": 1, "muestras.embedding": 1, "muestras.version_embedding": 1}
    for doc in _speakers().find({}, proyeccion):
        for m in doc.get("muestras", []):
            try:
                resultados.append((doc["nombre"], _descifrar_embedding(m["embedding"]), m.get("version_embedding", 0)))
            except (InvalidToken, ValueError, KeyError):
                log.warning("Embedding ilegible omitido (clave distinta o dato alterado)")
    return resultados


def eliminar_speaker_completo(nombre: str) -> tuple[bool, int]:
    """
    Elimina a una persona (documento + audios en disco). El documento se
    borra primero, de forma atómica; si luego falla el borrado de algún
    archivo, como mucho queda un audio huérfano cifrado, nunca un registro
    roto. Devuelve (existia, archivos_borrados).
    """
    doc = _speakers().find_one_and_delete({"nombre_clave": _clave(nombre)}, projection={"muestras.archivo_audio": 1})
    if doc is None:
        return False, 0
    return True, borrar_archivos_audio(m["archivo_audio"] for m in doc.get("muestras", []))


def renombrar_speaker(nombre_actual: str, nombre_nuevo: str) -> str:
    """Devuelve "ok", "no_existe" o "duplicado"."""
    clave_actual, clave_nueva = _clave(nombre_actual), _clave(nombre_nuevo)
    try:
        r = _speakers().update_one(
            {"nombre_clave": clave_actual},
            {"$set": {"nombre": nombre_nuevo.strip(), "nombre_clave": clave_nueva}},
        )
    except DuplicateKeyError:
        return "duplicado"
    return "ok" if r.matched_count == 1 else "no_existe"


# ---------------------------------------------------------------------------
# Estado anti fuerza bruta del PIN (persistente: sobrevive a reinicios)
# ---------------------------------------------------------------------------
def leer_estado_pin() -> dict:
    doc = _db()[COLECCION_SEGURIDAD].find_one({"_id": "pin"}) or {}
    return {
        "fallos": int(doc.get("fallos", 0)),
        "bloqueos": int(doc.get("bloqueos", 0)),
        "bloqueado_hasta": doc.get("bloqueado_hasta"),
    }


def guardar_estado_pin(fallos: int, bloqueos: int, bloqueado_hasta) -> None:
    _db()[COLECCION_SEGURIDAD].update_one(
        {"_id": "pin"},
        {"$set": {"fallos": int(fallos), "bloqueos": int(bloqueos), "bloqueado_hasta": bloqueado_hasta}},
        upsert=True,
    )
