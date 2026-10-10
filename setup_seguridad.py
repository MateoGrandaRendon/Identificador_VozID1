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

Si vuelves a correrlo, puedes conservar la clave de encriptación actual y
cambiar solo el PIN. Si generas una clave NUEVA, los embeddings y audios
ya guardados con la anterior quedarán ilegibles. Las variables
VOICE_ID_MONGODB_* que ya existieran en el .env se conservan.
"""

import getpass
import os
from pathlib import Path

from cryptography.fernet import Fernet

import security

ENV_PATH = Path(__file__).resolve().parent / ".env"


def main() -> None:
    print("=" * 60)
    print("  CONFIGURACIÓN DE SEGURIDAD — Sistema de Identificación de Voz")
    print("=" * 60)

    anteriores = {}
    if ENV_PATH.exists():
        for linea in ENV_PATH.read_text(encoding="utf-8").splitlines():
            clave, sep, valor = linea.partition("=")
            if sep:
                anteriores[clave.strip()] = valor.strip()

    clave_fernet = None
    if anteriores.get("VOICE_ID_FERNET_KEY"):
        conservar = input(
            "\nYa existe un archivo .env. ¿Conservar la clave de encriptación actual?\n"
            "   (s = solo cambiar el PIN, los datos guardados siguen legibles;\n"
            "    n = generar una clave NUEVA: los datos ya guardados quedarán ilegibles) (s/n): "
        ).strip().lower()
        if conservar == "s":
            clave_fernet = anteriores["VOICE_ID_FERNET_KEY"]
        elif input("⚠️  ¿Seguro que quieres perder los datos cifrados actuales? (escribe SI): ").strip() != "SI":
            print("Cancelado. No se modificó nada.")
            return

    print(
        f"\nDefine un PIN de acceso de {security.PIN_MIN_LARGO} a {security.PIN_MAX_LARGO} caracteres "
        "(no se mostrará en pantalla mientras lo escribes)."
    )
    while True:
        pin1 = getpass.getpass("Nuevo PIN: ").strip()
        pin2 = getpass.getpass("Confirma el PIN: ").strip()
        try:
            security.validar_pin_nuevo(pin1)
        except ValueError as e:
            print(f"❌ {e}")
            continue
        if pin1 != pin2:
            print("❌ No coinciden, intenta de nuevo.")
            continue
        break

    sal, hash_pin, iteraciones = security.generar_hash_pin(pin1)
    clave_fernet = clave_fernet or Fernet.generate_key().decode()

    # Se conservan las variables de MongoDB de un .env anterior (la URI puede
    # llevar credenciales que no deben perderse ni escribirse en el código).
    lineas = [
        f"VOICE_ID_PIN_SALT={sal}",
        f"VOICE_ID_PIN_HASH={hash_pin}",
        f"VOICE_ID_PIN_ITER={iteraciones}",
        f"VOICE_ID_FERNET_KEY={clave_fernet}",
    ] + [f"{k}={v}" for k, v in anteriores.items() if k.startswith("VOICE_ID_MONGODB_")]
    # Se escribe en un temporal creado YA con permisos 600 (nunca existe una copia legible por
    # otros usuarios, ni siquiera un instante) y luego se reemplaza el .env de forma atómica.
    # En Windows los permisos los dan las ACL de la carpeta del usuario (ya privadas).
    temporal = ENV_PATH.with_name(".env.tmp")
    temporal.unlink(missing_ok=True)
    fd = os.open(temporal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    os.replace(temporal, ENV_PATH)

    print(f"\n✅ Configuración guardada en: {ENV_PATH}")
    print("   Este archivo está en .gitignore — NUNCA debe subirse a GitHub.")
    print("   Guarda el PIN en un lugar seguro: si lo pierdes, no hay forma")
    print("   de recuperar los datos de voz ya encriptados con esta clave.")


if __name__ == "__main__":
    main()