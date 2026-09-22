"""
security.py
------------
Capa de seguridad del sistema:
  1) Control de acceso por PIN al iniciar el programa.
  2) Encriptación (Fernet / AES-128 autenticado) de los datos biométricos
     que se guardan en disco: embeddings (.npy) y audio (.wav).

Todo el material sensible (sal, hash del PIN, clave de encriptación) vive
en variables de entorno (.env), NUNCA en el código fuente ni en la base
de datos. Este módulo no lee ni escribe el .env directamente: eso lo hace
config.py (con python-dotenv) y setup_seguridad.py (al crearlo).
"""

import hashlib
import secrets
import time

from cryptography.fernet import Fernet

import config

ITERACIONES_PBKDF2 = 200_000   # coste del hash: alto para frenar fuerza bruta,
                                 # bajo lo suficiente para no notar demora al escribir el PIN
MAX_INTENTOS = 3
BLOQUEO_SEGUNDOS = 60


# ---------------------------------------------------------------------------
# PIN de acceso
# ---------------------------------------------------------------------------
def _hash_pin(pin: str, sal: str) -> str:
    """PBKDF2-HMAC-SHA256: función de hash lenta a propósito (resiste fuerza bruta)."""
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), sal.encode(), ITERACIONES_PBKDF2).hex()


def generar_hash_pin(pin: str) -> tuple[str, str]:
    """
    Genera (sal, hash) para un PIN nuevo. Se usa una sola vez, desde
    setup_seguridad.py — nunca se llama en el flujo normal del programa.
    """
    sal = secrets.token_hex(16)
    hash_pin = _hash_pin(pin, sal)
    return sal, hash_pin


def verificar_pin(pin_ingresado: str, sal: str, hash_guardado: str) -> bool:
    """
    Compara el PIN ingresado contra el hash guardado.
    Usa comparación en tiempo constante (secrets.compare_digest) para que
    el tiempo de respuesta no revele por dónde falla la comparación
    (timing attack) — una comparación normal con '==' sí sería vulnerable.
    """
    hash_calculado = _hash_pin(pin_ingresado, sal)
    return secrets.compare_digest(hash_calculado, hash_guardado)


def pedir_acceso() -> bool:
    """
    Pide el PIN de acceso al iniciar el programa. Devuelve True si el
    acceso fue concedido. Bloquea temporalmente tras varios intentos
    fallidos (mitiga fuerza bruta interactiva).
    """
    if not config.PIN_SALT or not config.PIN_HASH:
        print("⚠️  No hay un PIN de acceso configurado todavía.")
        print("   Ejecuta primero:  python setup_seguridad.py")
        return False

    intentos = 0
    while intentos < MAX_INTENTOS:
        pin = input("🔒 PIN de acceso: ").strip()
        if verificar_pin(pin, config.PIN_SALT, config.PIN_HASH):
            print("✅ Acceso concedido.\n")
            return True
        intentos += 1
        print(f"❌ PIN incorrecto ({intentos}/{MAX_INTENTOS})")

    print(f"🔒 Demasiados intentos fallidos. Programa bloqueado {BLOQUEO_SEGUNDOS}s.")
    time.sleep(BLOQUEO_SEGUNDOS)
    return False


# ---------------------------------------------------------------------------
# Encriptación de datos biométricos (embeddings y audio en disco)
# ---------------------------------------------------------------------------
def _obtener_fernet() -> Fernet:
    if not config.FERNET_KEY:
        raise RuntimeError(
            "No hay clave de encriptación configurada (VOICE_ID_FERNET_KEY en .env). "
            "Ejecuta primero: python setup_seguridad.py"
        )
    return Fernet(config.FERNET_KEY.encode())


def encriptar_bytes(datos: bytes) -> bytes:
    """Encripta bytes (AES-128 autenticado vía Fernet) antes de guardarlos en disco."""
    return _obtener_fernet().encrypt(datos)


def desencriptar_bytes(datos_encriptados: bytes) -> bytes:
    """
    Desencripta bytes leídos desde disco. Lanza cryptography.fernet.InvalidToken
    si la clave no coincide o si el archivo fue alterado — Fernet incluye
    autenticación (HMAC), así que además de confidencialidad detecta manipulación.
    """
    return _obtener_fernet().decrypt(datos_encriptados)