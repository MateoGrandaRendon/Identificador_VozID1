"""
setup_seguridad.py
-------------------
Script de configuración inicial de seguridad. Se ejecuta UNA sola vez
(o cada vez que quieras cambiar el PIN / rotar la clave) para generar:
  - la sal y el hash del PIN de acceso
  - la clave de encriptación (Fernet) de los datos biométricos
y guardarlos en el archivo .env, que NUNCA se sube a GitHub (ya está
en .gitignore).

Uso (con el entorno virtual activado):
    python setup_seguridad.py

⚠️  IMPORTANTE: si vuelves a correr este script, la clave de encriptación
anterior se pierde y los archivos .npy/.wav ya guardados con esa clave
quedarán ilegibles. Guarda el PIN en un lugar seguro.
"""

import getpass
from pathlib import Path

from cryptography.fernet import Fernet

import security

ENV_PATH = Path(__file__).resolve().parent / ".env"


def main() -> None:
    print("=" * 60)
    print("  CONFIGURACIÓN DE SEGURIDAD — Sistema de Identificación de Voz")
    print("=" * 60)

    if ENV_PATH.exists():
        confirmar = input(
            "\n⚠️  Ya existe un archivo .env. Continuar lo SOBREESCRIBIRÁ: el PIN\n"
            "   y la clave de encriptación viejos dejarán de funcionar, y los\n"
            "   datos ya encriptados con la clave anterior quedarán ilegibles.\n"
            "   ¿Continuar de todas formas? (s/n): "
        ).strip().lower()
        if confirmar != "s":
            print("Cancelado. No se modificó nada.")
            return

    print("\nDefine un PIN de acceso (no se mostrará en pantalla mientras lo escribes).")
    while True:
        pin1 = getpass.getpass("Nuevo PIN: ").strip()
        pin2 = getpass.getpass("Confirma el PIN: ").strip()
        if not pin1:
            print("❌ El PIN no puede estar vacío.")
            continue
        if pin1 != pin2:
            print("❌ No coinciden, intenta de nuevo.")
            continue
        break

    sal, hash_pin = security.generar_hash_pin(pin1)
    clave_fernet = Fernet.generate_key().decode()

    contenido = (
        f"VOICE_ID_PIN_SALT={sal}\n"
        f"VOICE_ID_PIN_HASH={hash_pin}\n"
        f"VOICE_ID_FERNET_KEY={clave_fernet}\n"
    )
    ENV_PATH.write_text(contenido, encoding="utf-8")

    print(f"\n✅ Configuración guardada en: {ENV_PATH}")
    print("   Este archivo está en .gitignore — NUNCA debe subirse a GitHub.")
    print("   Guarda el PIN en un lugar seguro: si lo pierdes, no hay forma")
    print("   de recuperar los datos de voz ya encriptados con esta clave.")


if __name__ == "__main__":
    main()