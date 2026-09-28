
"""Arranque de la interfaz gráfica.  Uso (desde la raíz del proyecto, con .venv activo):
    python -m frontend.app
"""
import threading
from pathlib import Path

import webview

import audio_processor as ap
from frontend.bridge import Api

WEB = Path(__file__).resolve().parent / "web"
REQUERIDOS = ("index.html", "styles.css", "app.js")


def _verificar_frontend() -> None:
    faltan = [n for n in REQUERIDOS if not (WEB / n).is_file()]
    if faltan:
        raise SystemExit(
            f"❌ Faltan archivos del frontend.\n   Carpeta buscada: {WEB}\n   Faltan: {', '.join(faltan)}\n"
            "   Estructura esperada:\n     frontend/app.py, bridge.py, __init__.py\n"
            "     frontend/web/index.html, styles.css, app.js"
        )


def main() -> None:
    _verificar_frontend()
    api = Api()
    ventana = webview.create_window(
        "VoiceID · Identificación biométrica por voz", str(WEB / "index.html"),
        js_api=api, width=1280, height=820, min_size=(900, 620), background_color="#070b14",
    )
    api.set_window(ventana)
    ventana.events.closing += api.shutdown  # libera el micrófono también al cerrar con la X
    # Precalienta librosa/numba en segundo plano, como hace el CLI al iniciar
    webview.start(lambda: threading.Thread(target=ap.precalentar_motor, daemon=True).start())
    api.shutdown()


if __name__ == "__main__":
    main()
