"""
security.py
------------
Capa de seguridad del sistema:
  1) Control de acceso por PIN al iniciar el programa, con bloqueo
     progresivo y PERSISTENTE ante intentos fallidos (anti fuerza bruta).
  2) Encriptación (Fernet / AES-128 autenticado) de los datos biométricos:
     embeddings (en MongoDB) y audio (.wav en disco).
  3) Validación y sanitización de las entradas del usuario.
  4) Registro técnico de errores en archivo, sin datos personales.

Todo el material sensible (sal, hash del PIN, clave de encriptación,
cadena de conexión a MongoDB) vive en variables de entorno (.env), NUNCA en
el código fuente ni en la base de datos.
"""

import datetime
import getpass
import hashlib
import logging
import re
import secrets
import threading
import time
import unicodedata
from logging.handlers import RotatingFileHandler

from cryptography.fernet import Fernet

import config

log = logging.getLogger("voiceid.security")

ITERACIONES_PBKDF2 = 600_000   # para PINes NUEVOS (OWASP 2023); los antiguos usan config.PIN_ITERACIONES
PIN_MIN_LARGO = 6
PIN_MAX_LARGO = 64
MAX_INTENTOS = 3
BLOQUEO_SEGUNDOS = 60          # primer bloqueo; se duplica en cada bloqueo consecutivo
BLOQUEO_MAX_SEGUNDOS = 60 * 60


# ---------------------------------------------------------------------------
# Registro de errores (logs)
# ---------------------------------------------------------------------------
def configurar_logs() -> None:
    """
    Envía los errores técnicos a data/logs/voiceid.log (rotativo, 1 MB x 3)
    en vez de mostrarlos al usuario. Los mensajes del proyecto nunca
    incluyen nombres de personas, PINes, claves ni la URI de MongoDB.
    """
    raiz = logging.getLogger("voiceid")
    if raiz.handlers:
        return
    manejador = RotatingFileHandler(config.LOG_DIR / "voiceid.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    manejador.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    raiz.addHandler(manejador)
    raiz.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Validación de entradas
# ---------------------------------------------------------------------------
# Letras (de cualquier idioma), dígitos, espacios y . ' - ; debe empezar por letra.
_PATRON_NOMBRE = re.compile(r"^[^\W\d_][\w .'\-]*$", re.UNICODE)


def validar_nombre_persona(nombre) -> str:
    """
    Devuelve el nombre normalizado (NFC, espacios colapsados) o lanza
    ValueError con un mensaje apto para el usuario. Rechaza cualquier tipo
    que no sea texto: así un objeto como {"$ne": ""} enviado desde la
    interfaz nunca llega a la base de datos (inyección NoSQL).
    """
    if not isinstance(nombre, str):
        raise ValueError("El nombre debe ser texto.")
    nombre = " ".join(unicodedata.normalize("NFC", nombre).split())
    if not nombre:
        raise ValueError("El nombre no puede estar vacío.")
    if len(nombre) > config.NOMBRE_MAX_LARGO:
        raise ValueError(f"El nombre no puede superar {config.NOMBRE_MAX_LARGO} caracteres.")
    if not _PATRON_NOMBRE.fullmatch(nombre):
        raise ValueError("El nombre solo puede contener letras, números, espacios y . ' - (y debe empezar por una letra).")
    return nombre


def validar_pin_nuevo(pin: str) -> None:
    if not isinstance(pin, str) or not (PIN_MIN_LARGO <= len(pin) <= PIN_MAX_LARGO):
        raise ValueError(f"El PIN debe tener entre {PIN_MIN_LARGO} y {PIN_MAX_LARGO} caracteres.")
    if len(set(pin)) == 1 or pin in "0123456789012345" or pin in "9876543210987654":
        raise ValueError("El PIN es demasiado predecible (repetido o secuencial).")


# ---------------------------------------------------------------------------
# PIN de acceso
# ---------------------------------------------------------------------------
def _hash_pin(pin: str, sal: str, iteraciones: int) -> str:
    """PBKDF2-HMAC-SHA256: función de hash lenta a propósito (resiste fuerza bruta)."""
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), sal.encode(), iteraciones).hex()


def generar_hash_pin(pin: str) -> tuple[str, str, int]:
    """
    Genera (sal, hash, iteraciones) para un PIN nuevo. Se usa desde
    setup_seguridad.py — nunca en el flujo normal del programa.
    """
    sal = secrets.token_hex(16)
    return sal, _hash_pin(pin, sal, ITERACIONES_PBKDF2), ITERACIONES_PBKDF2


def verificar_pin(pin_ingresado, sal: str, hash_guardado: str, iteraciones: int = None) -> bool:
    """
    Compara el PIN ingresado contra el hash guardado en tiempo constante
    (secrets.compare_digest), para que el tiempo de respuesta no revele
    por dónde falla la comparación (timing attack).
    """
    if not isinstance(pin_ingresado, str) or len(pin_ingresado) > PIN_MAX_LARGO:
        return False
    hash_calculado = _hash_pin(pin_ingresado, sal, iteraciones or config.PIN_ITERACIONES)
    return secrets.compare_digest(hash_calculado, hash_guardado)


class ControlAcceso:
    """
    Verificación del PIN con bloqueo progresivo: tras MAX_INTENTOS fallos
    se bloquea BLOQUEO_SEGUNDOS, y cada bloqueo consecutivo duplica la
    espera (60 s, 120 s, 240 s… hasta 1 h). El estado se guarda en MongoDB,
    así que cerrar y volver a abrir el programa NO reinicia el contador.
    Si la base de datos no responde, se mantiene en memoria.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._memoria = {"fallos": 0, "bloqueos": 0, "bloqueado_hasta": None}

    def _leer(self) -> dict:
        import database as db  # import diferido: database ya importa este módulo
        try:
            self._memoria = db.leer_estado_pin()
        except Exception:
            log.warning("Estado anti fuerza bruta no disponible en la BD; se usa memoria")
        return dict(self._memoria)

    def _guardar(self, estado: dict) -> None:
        import database as db
        self._memoria = dict(estado)
        try:
            db.guardar_estado_pin(**estado)
        except Exception:
            log.warning("No se pudo persistir el estado anti fuerza bruta")

    def segundos_bloqueo(self) -> int:
        hasta = self._leer()["bloqueado_hasta"]
        if hasta is None:
            return 0
        if hasta.tzinfo is None:
            hasta = hasta.replace(tzinfo=datetime.timezone.utc)
        restante = (hasta - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        return max(0, int(restante) + 1) if restante > 0 else 0

    def intentar(self, pin) -> tuple[str, int]:
        """
        Devuelve (resultado, dato):
          ("ok", 0) · ("bloqueado", segundos) · ("incorrecto", intentos_restantes)
        """
        with self._lock:
            espera = self.segundos_bloqueo()
            if espera:
                return "bloqueado", espera
            estado = self._leer()
            if verificar_pin(pin, config.PIN_SALT, config.PIN_HASH):
                self._guardar({"fallos": 0, "bloqueos": 0, "bloqueado_hasta": None})
                return "ok", 0
            estado["fallos"] += 1
            log.warning("PIN incorrecto (%d/%d)", estado["fallos"], MAX_INTENTOS)
            if estado["fallos"] >= MAX_INTENTOS:
                segundos = min(BLOQUEO_MAX_SEGUNDOS, BLOQUEO_SEGUNDOS * 2 ** estado["bloqueos"])
                estado.update(
                    fallos=0,
                    bloqueos=estado["bloqueos"] + 1,
                    bloqueado_hasta=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=segundos),
                )
                self._guardar(estado)
                return "bloqueado", segundos
            self._guardar(estado)
            return "incorrecto", MAX_INTENTOS - estado["fallos"]


def pin_configurado() -> bool:
    return bool(config.PIN_SALT and config.PIN_HASH)


def pedir_acceso() -> bool:
    """Pide el PIN en la terminal (sin mostrarlo). Devuelve True si el acceso fue concedido."""
    if not pin_configurado():
        print("⚠️  No hay un PIN de acceso configurado todavía.")
        print("   Ejecuta primero:  python setup_seguridad.py")
        return False

    control = ControlAcceso()
    while True:
        espera = control.segundos_bloqueo()
        if espera:
            print(f"🔒 Acceso bloqueado por intentos fallidos. Inténtalo de nuevo en {espera} s.")
            return False
        resultado, dato = control.intentar(getpass.getpass("🔒 PIN de acceso: ").strip())
        if resultado == "ok":
            print("✅ Acceso concedido.\n")
            return True
        if resultado == "bloqueado":
            print(f"🔒 Demasiados intentos fallidos. Bloqueado {dato} s.")
            return False
        print(f"❌ PIN incorrecto (quedan {dato} intento(s))")
        time.sleep(1)  # frena intentos automatizados


# ---------------------------------------------------------------------------
# Encriptación de datos biométricos (embeddings y audio)
# ---------------------------------------------------------------------------
def _obtener_fernet() -> Fernet:
    if not config.FERNET_KEY:
        raise RuntimeError(
            "No hay clave de encriptación configurada (VOICE_ID_FERNET_KEY en .env). "
            "Ejecuta primero: python setup_seguridad.py"
        )
    return Fernet(config.FERNET_KEY.encode())


def verificar_clave_cifrado() -> None:
    """
    Se llama al arrancar: una clave ausente o mal copiada en el .env se detecta
    aquí con un mensaje claro, en vez de que todos los embeddings se omitan en
    silencio («no hay personas registradas») o falle el primer registro.
    """
    try:
        _obtener_fernet()
    except ValueError:
        raise RuntimeError(
            "VOICE_ID_FERNET_KEY en .env no es una clave válida (¿se copió incompleta?). "
            "Restaura la clave original: si generas otra, los datos ya cifrados quedarán ilegibles."
        ) from None


def encriptar_bytes(datos: bytes) -> bytes:
    """Encripta bytes (AES-128 autenticado vía Fernet)."""
    return _obtener_fernet().encrypt(datos)


def desencriptar_bytes(datos_encriptados: bytes) -> bytes:
    """
    Desencripta bytes. Lanza cryptography.fernet.InvalidToken si la clave no
    coincide o si el dato fue alterado — Fernet incluye autenticación (HMAC),
    así que además de confidencialidad detecta manipulación.
    """
    return _obtener_fernet().decrypt(datos_encriptados)
