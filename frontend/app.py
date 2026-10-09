
"""Arranque de la interfaz gráfica.  Uso (desde la raíz del proyecto, con .venv activo):
    python -m frontend.app

Endurecimiento de la ventana:
  - La página se entrega como HTML en memoria (CSS y JS incrustados), NO
    desde un archivo local. Con un archivo, pywebview levanta su propio
    servidor HTTP en 127.0.0.1 (sin TLS, sin cabeceras de seguridad y con un
    endpoint con CORS "*"); así no hay ningún puerto abierto.
  - Content-Security-Policy con un nonce aleatorio por arranque: solo el
    script incrustado por esta función puede ejecutarse; nada se carga de
    Internet, no hay formularios hacia fuera ni marcos (clickjacking).
  - Sin herramientas de desarrollo (debug=False), sin acceso a file:// y
    modo privado (no se guarda caché ni almacenamiento del navegador).
  - Si la ventana llegara a navegar a otra página, la sesión se bloquea.
"""
import logging
import secrets
import sys
import threading
from pathlib import Path

import webview

import audio_processor as ap
import database as db
import security
from frontend.bridge import Api

log = logging.getLogger("voiceid.app")

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


def _construir_html() -> str:
    """index.html con styles.css y app.js incrustados y la CSP con un nonce nuevo."""
    nonce = secrets.token_urlsafe(18)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    # 'unsafe-eval': pywebview construye window.pywebview.api con new Function();
    #   no permite ejecutar <script> ni manejadores inline inyectados (siguen
    #   exigiendo el nonce), y app.js nunca evalúa texto.
    # style-src 'unsafe-inline': pywebview inserta sus propios <style>.
    csp = (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}' 'unsafe-eval'; "
        "style-src 'unsafe-inline'; "
        "img-src data:; font-src 'none'; connect-src 'none'; media-src 'none'; "
        "object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'"
    )
    marcadores = {
        '<link rel="stylesheet" href="styles.css">': f'<style nonce="{nonce}">{css}</style>',
        '<script src="app.js"></script>': f'<script nonce="{nonce}">{js}</script>',
        "<!--CSP-->": f'<meta http-equiv="Content-Security-Policy" content="{csp}">',
    }
    for marcador, contenido in marcadores.items():
        if marcador not in html:
            raise SystemExit(f"❌ index.html no contiene el marcador esperado: {marcador}")
        html = html.replace(marcador, contenido, 1)
    return html


def main() -> None:
    security.configurar_logs()
    _verificar_frontend()
    try:
        api = Api()
    except db.BaseDatosError as e:
        print(f"❌ {e}")
        sys.exit(1)

    webview.settings["ALLOW_FILE_URLS"] = False           # sin --allow-file-access-from-files
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    webview.settings["ALLOW_DOWNLOADS"] = False

    ventana = webview.create_window(
        "VoiceID · Identificación biométrica por voz", html=_construir_html(),
        js_api=api, width=1280, height=820, min_size=(900, 620), background_color="#070b14",
    )
    api._set_window(ventana)

    def _proteger_navegacion():
        # La página propia no tiene URL http(s); si aparece una, alguien navegó fuera de la app.
        url = (ventana.get_current_url() or "").lower()
        if url.startswith(("http:", "https:", "file:")):
            log.warning("La ventana navegó a una página externa: sesión bloqueada")
            api._cerrar_sesion()
            ventana.load_html(_construir_html())

    ventana.events.loaded += _proteger_navegacion
    ventana.events.closing += api._shutdown  # libera el micrófono también al cerrar con la X
    # Precalienta librosa/numba en segundo plano, como hace el CLI al iniciar
    webview.start(lambda: threading.Thread(target=ap.precalentar_motor, daemon=True).start(),
                  debug=False, private_mode=True)
    api._shutdown()


if __name__ == "__main__":
    main()
