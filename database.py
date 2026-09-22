"""
database.py
-----------
Capa de acceso a datos (ORM con SQLAlchemy sobre SQLite).

Por qué esta elección:
 - SQLite no requiere instalar ni administrar un servidor: ideal para esta
   Fase 1 / prototipo local que corre desde VS Code.
 - SQLAlchemy desacopla la lógica de negocio del motor de base de datos:
   si más adelante el proyecto crece (por ejemplo, para la interfaz web),
   se puede migrar a PostgreSQL/MySQL cambiando solo la cadena de conexión.
 - Los embeddings (vectores numpy que representan la voz) se guardan como
   archivos .npy en disco, y la base de datos solo almacena la RUTA a ese
   archivo. Esto evita guardar BLOBs pesados en SQLite y facilita depurar
   o inspeccionar un embedding de forma independiente.

Modelo de datos:
    Speaker (persona registrada)
        └── VoiceSample (una muestra de voz: audio + embedding) [1 a N]
"""

import datetime
import io
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

import config
import security

Base = declarative_base()


class Speaker(Base):
    """Representa a una persona registrada en el sistema."""

    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(120), nullable=False, unique=True)
    fecha_registro = Column(DateTime, default=datetime.datetime.utcnow)

    # Si se borra un Speaker, se borran también sus muestras (cascade)
    muestras = relationship(
        "VoiceSample", back_populates="speaker", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Speaker id={self.id} nombre='{self.nombre}'>"


class VoiceSample(Base):
    """Una muestra de voz individual (audio + embedding) de un Speaker."""

    __tablename__ = "voice_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=False)
    ruta_audio = Column(String(300), nullable=False)       # archivo .wav
    ruta_embedding = Column(String(300), nullable=False)   # archivo .npy
    fecha_creacion = Column(DateTime, default=datetime.datetime.utcnow)

    speaker = relationship("Speaker", back_populates="muestras")

    def __repr__(self):
        return f"<VoiceSample id={self.id} speaker_id={self.speaker_id}>"


# ---------------------------------------------------------------------------
# Motor de base de datos y fábrica de sesiones
# ---------------------------------------------------------------------------
engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Crea las tablas en la base de datos si todavía no existen."""
    Base.metadata.create_all(engine)


def obtener_o_crear_speaker(session, nombre: str) -> Speaker:
    """Busca un Speaker por nombre; si no existe, lo crea."""
    speaker = session.query(Speaker).filter_by(nombre=nombre).first()
    if speaker is None:
        speaker = Speaker(nombre=nombre)
        session.add(speaker)
        session.commit()
        session.refresh(speaker)
    return speaker


def guardar_muestra(session, speaker: Speaker, ruta_audio: str, embedding: np.ndarray) -> VoiceSample:
    """
    Guarda el embedding ENCRIPTADO en disco (mismo nombre base que el audio,
    con extensión .npy.enc) y registra la referencia (metadatos) en la BD.

    El embedding es un dato biométrico (una "huella" numérica de la voz de
    la persona), así que nunca se escribe en texto plano: se serializa con
    numpy a un buffer en memoria, se encripta con Fernet (AES-128
    autenticado) y solo el resultado encriptado toca el disco.
    """
    ruta_embedding = str(ruta_audio).replace(".wav.enc", ".npy.enc").replace(".wav", ".npy.enc")

    buffer = io.BytesIO()
    np.save(buffer, embedding)
    datos_encriptados = security.encriptar_bytes(buffer.getvalue())
    Path(ruta_embedding).write_bytes(datos_encriptados)

    muestra = VoiceSample(
        speaker_id=speaker.id,
        ruta_audio=str(ruta_audio),
        ruta_embedding=ruta_embedding,
    )
    session.add(muestra)
    session.commit()
    session.refresh(muestra)
    return muestra


def listar_speakers(session):
    """Devuelve todos los Speakers registrados."""
    return session.query(Speaker).all()


def obtener_embeddings_todos(session):
    """
    Carga y DESENCRIPTA todos los embeddings guardados en disco, y los
    devuelve como una lista de tuplas (nombre_speaker, embedding_numpy).
    Se usa para comparar una voz nueva contra todos los perfiles registrados.
    """
    resultados = []
    for muestra in session.query(VoiceSample).all():
        try:
            datos_encriptados = Path(muestra.ruta_embedding).read_bytes()
            datos_planos = security.desencriptar_bytes(datos_encriptados)
            embedding = np.load(io.BytesIO(datos_planos))
            resultados.append((muestra.speaker.nombre, embedding))
        except FileNotFoundError:
            # Si el archivo .npy.enc fue borrado manualmente, se ignora esa muestra
            continue
    return resultados


def eliminar_speaker_completo(nombre: str) -> tuple[bool, int]:
    """
    Elimina un speaker de forma QUIRÚRGICA Y DIRECTA:

      - Usa SQL parametrizado (text()) en vez del ORM completo, evitando
        que SQLAlchemy cargue objetos Python de más (sin "hidratar" el
        Speaker ni sus VoiceSample como instancias del ORM).
      - Todo ocurre en UNA SOLA transacción atómica ligera: `engine.begin()`
        abre la transacción y hace commit() automático al salir del bloque
        `with` (o rollback si algo falla) — no hay pasos intermedios.
      - NO genera ningún respaldo (backup) de la base de datos ni vuelca
        tablas completas: solo lee las 2 rutas de archivo que necesita
        borrar y ejecuta 2 DELETE puntuales por clave. Es una operación
        O(muestras de esa persona), nunca O(tamaño total de la BD).

    Los archivos físicos (.wav.enc/.npy.enc) se borran DESPUÉS de que la
    transacción de la BD ya fue confirmada (commit) — así, si el borrado
    de un archivo falla (permisos, ya no existe), la base de datos queda
    consistente de todas formas; en el peor caso queda un archivo huérfano
    en disco, nunca un registro roto.

    Devuelve (existia, archivos_borrados).
    """
    with engine.begin() as conexion:  # transacción única: commit automático al salir, rollback si hay error
        fila_speaker = conexion.execute(
            text("SELECT id FROM speakers WHERE nombre = :nombre"),
            {"nombre": nombre},
        ).first()

        if fila_speaker is None:
            return False, 0  # no existe: no se abre ninguna operación de escritura

        speaker_id = fila_speaker.id

        filas_muestras = conexion.execute(
            text("SELECT ruta_audio, ruta_embedding FROM voice_samples WHERE speaker_id = :id"),
            {"id": speaker_id},
        ).all()

        # Dos DELETE puntuales y parametrizados, dentro de la MISMA transacción:
        conexion.execute(
            text("DELETE FROM voice_samples WHERE speaker_id = :id"), {"id": speaker_id}
        )
        conexion.execute(
            text("DELETE FROM speakers WHERE id = :id"), {"id": speaker_id}
        )
    # --- fin del `with engine.begin()`: aquí ya se hizo commit() atómico ---

    borrados = 0
    for fila in filas_muestras:
        for ruta in (fila.ruta_audio, fila.ruta_embedding):
            try:
                Path(ruta).unlink(missing_ok=True)
                borrados += 1
            except OSError:
                pass  # el archivo ya no existía o no hay permisos: no es un error fatal

    return True, borrados


def eliminar_muestras_speaker(session, speaker: Speaker) -> int:
    """
    Elimina TODAS las muestras (audio + embedding, en BD y en disco) de un
    Speaker que YA EXISTE, sin borrar al Speaker en sí. Se usa para
    "re-entrenar": vaciar las muestras viejas antes de grabar unas nuevas.
    Devuelve cuántas muestras se eliminaron.
    """
    muestras = list(speaker.muestras)
    for muestra in muestras:
        for ruta in (muestra.ruta_audio, muestra.ruta_embedding):
            try:
                Path(ruta).unlink(missing_ok=True)
            except OSError:
                pass
        session.delete(muestra)
    session.commit()
    return len(muestras)


def renombrar_speaker(session, nombre_actual: str, nombre_nuevo: str) -> str:
    """
    Cambia el nombre de un Speaker existente.

    Devuelve:
        "ok"        si se renombró correctamente
        "no_existe" si nombre_actual no está registrado
        "duplicado" si nombre_nuevo ya lo usa otra persona
    """
    if nombre_actual == nombre_nuevo:
        return "ok"

    speaker = session.query(Speaker).filter_by(nombre=nombre_actual).first()
    if speaker is None:
        return "no_existe"

    ya_existe = session.query(Speaker).filter_by(nombre=nombre_nuevo).first()
    if ya_existe is not None:
        return "duplicado"

    speaker.nombre = nombre_nuevo
    session.commit()
    return "ok"