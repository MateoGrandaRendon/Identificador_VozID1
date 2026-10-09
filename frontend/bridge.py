"""
bridge.py — capa de comunicación Frontend ⇄ Backend (expuesta a JS vía pywebview).

Mapa UI → backend existente (no se duplica lógica, solo se orquesta):
  Registrar / Re-entrenar /
  Añadir muestras         : ap.validar_calidad_muestra, registro.guardar_perfil
  Identificar             : matching.filtrar_perfiles_compatibles / identificar_mejor_candidato,
                            ap.extraer_embedding, ap.calcular_puntaje_vivacidad
  Eliminar                : db.eliminar_speaker_completo
  Gestionar               : db.listar_speakers, db.renombrar_speaker
  Análisis de voz         : ap._estimar_f0_bloque, ap._calcular_volumen_rms, ap.clasificar_tono
  Pruebas                 : ap.grabar_audio, cli.probar_base_datos, ap.extraer_embedding, ap.similitud_coseno
  PIN                     : security.ControlAcceso (mismo PIN/.env y mismo bloqueo que el CLI)

Superficie de ataque: pywebview expone a JavaScript TODOS los métodos
públicos de Api. Por eso:
  - Todo lo que no debe llamarse desde JS empieza por "_".
  - Todo método público pasa por @_safe: exige sesión (salvo _ABIERTOS),
    limita la frecuencia de llamadas, valida tipos y nunca devuelve trazas
    ni mensajes internos (van al log).
  - La sesión caduca por inactividad y por tiempo máximo (config.SESION_*).

Limitación: el backend graba con duración fija y bloqueante (grabar_audio*), incompatible con
Iniciar/Detener y niveles en vivo. Por eso _Recorder abre su propio sd.InputStream, pero reutiliza
las funciones de análisis del backend. Todo método devuelve {ok, data, error, code}.
"""
import collections
import functools
import inspect
import logging
import math
import random
import threading
import time

import numpy as np
import sounddevice as sd

import audio_processor as ap
import config
import database as db
import main as cli  # solo se reutilizan helpers; importar main no ejecuta nada
import matching
import registro
import security

log = logging.getLogger("voiceid.bridge")

_BLOQUE = int(0.1 * config.SAMPLE_RATE)                              # 100 ms: nivel/onda fluidos
_VENTANA_F0 = int(config.BLOQUE_ANALISIS_SEG * config.SAMPLE_RATE)   # 300 ms, igual que el CLI
_MAX_MUESTRAS_AUDIO = config.GRABACION_MAX_SEG * config.SAMPLE_RATE
_ABIERTOS = {"auth_status", "login", "get_config"}

# Límite de llamadas por método: (máximo, ventana en segundos). El resto usa _LIMITE_DEFECTO.
_LIMITES = {
    "get_live": (120, 10),          # la UI consulta 10 veces/s durante una grabación
    "login": (5, 10),               # además del bloqueo progresivo por PIN incorrecto
    "wizard_finish": (6, 60),       # operaciones pesadas (extracción de embeddings)
    "identify_run": (12, 60),
    "run_test": (9, 60),
    "delete_speaker": (10, 60),
}
_LIMITE_DEFECTO = (60, 10)
_MENSAJE_INTERNO = "Error interno. Los detalles técnicos se guardaron en data/logs/voiceid.log."


class ApiError(Exception):
    def __init__(self, msg, code="error"):
        super().__init__(msg)
        self.code = code


class _Limitador:
    """Ventana deslizante por método: frena abusos y bucles que saturen CPU/BD (DoS)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._llamadas = collections.defaultdict(collections.deque)

    def permitir(self, nombre: str) -> bool:
        maximo, ventana = _LIMITES.get(nombre, _LIMITE_DEFECTO)
        ahora = time.monotonic()
        with self._lock:
            cola = self._llamadas[nombre]
            while cola and ahora - cola[0] > ventana:
                cola.popleft()
            if len(cola) >= maximo:
                return False
            cola.append(ahora)
            return True


def _safe(fn):
    firma = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        nombre = fn.__name__
        if not self._limitador.permitir(nombre):
            return {"ok": False, "data": None, "error": "Demasiadas solicitudes: espera unos segundos.", "code": "rate_limited"}
        if nombre not in _ABIERTOS:
            motivo = self._sesion_invalida()
            if motivo:
                return {"ok": False, "data": None, "error": motivo, "code": "auth"}
            self._ultima_actividad = time.monotonic()
        # Llamada desde JS con argumentos de más/de menos: se detecta ANTES de ejecutar, para
        # no confundirla con un TypeError interno (que es un bug y va como backend_error).
        try:
            firma.bind(self, *a, **k)
        except TypeError:
            log.warning("Llamada con argumentos inválidos a %s", nombre)
            return {"ok": False, "data": None, "error": "Solicitud inválida.", "code": "invalid"}
        try:
            return {"ok": True, "data": fn(self, *a, **k), "error": None, "code": None}
        except ApiError as e:
            return {"ok": False, "data": None, "error": str(e), "code": e.code}
        except db.BaseDatosError as e:   # mensajes ya redactados para el usuario
            return {"ok": False, "data": None, "error": str(e), "code": "db"}
        except Exception:  # nunca dejar que una excepción rompa el puente ni exponer trazas
            log.exception("Error no controlado en %s", nombre)
            return {"ok": False, "data": None, "error": _MENSAJE_INTERNO, "code": "backend_error"}
    return wrapper


def _nombre(valor) -> str:
    try:
        return security.validar_nombre_persona(valor)
    except ValueError as e:
        raise ApiError(str(e), "invalid") from None


class _Recorder:
    """Captura Iniciar/Detener con nivel (RMS/dBFS), F0 y forma de onda REALES del micrófono."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stream = None
        self._reset()

    def _reset(self):
        self.active, self.t0 = False, 0.0
        self.blocks, self.vols, self.f0s = [], [], []
        self.rms, self.f0, self.wave, self.n = 0.0, None, [], 0
        self.total = 0
        self.ring = np.zeros(0, dtype=np.float32)

    def procesar(self, b):
        rms = ap._calcular_volumen_rms(b)
        with self._lock:
            if self.total >= _MAX_MUESTRAS_AUDIO:
                return  # tope duro: la memoria no crece aunque la UI nunca pulse "Detener"
            self.blocks.append(np.array(b, dtype=np.float32))
            self.total += len(b)
            self.vols.append(rms)
            self.rms = rms
            self.wave = np.asarray(b[:: max(1, len(b) // 64)][:64], dtype=float).tolist()
            self.ring = np.concatenate([self.ring, b])[-_VENTANA_F0:]
            self.n += 1
            if self.n % 3 == 0:
                self.f0 = ap._estimar_f0_bloque(self.ring, config.SAMPLE_RATE)
                if self.f0:
                    self.f0s.append(self.f0)

    def start(self):
        # Comprobar y reservar bajo el mismo lock: pywebview atiende cada llamada de JS en
        # su propio hilo, y dos "Iniciar" seguidos no deben abrir dos streams.
        with self._lock:
            if self.active:
                raise ApiError("Ya hay una grabación en curso.", "busy")
            self._reset()
            self.active, self.t0 = True, time.time()
        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, dtype="float32",
                blocksize=_BLOQUE, callback=lambda indata, f, t, s: self.procesar(indata[:, 0]),
            )
            self._stream.start()
        except Exception:
            self._stream = None
            self.active = False
            log.exception("No se pudo abrir el micrófono")
            raise ApiError("Micrófono no disponible. Revisa que esté conectado y con permisos.", "mic_unavailable")

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.active = False
        with self._lock:
            return np.concatenate(self.blocks) if self.blocks else np.zeros(0, dtype=np.float32)

    def live(self):
        with self._lock:
            rms, f0, wave = self.rms, self.f0, list(self.wave)
        return {
            "recording": self.active, "rms": rms, "wave": wave, "f0": f0,
            "db": 20 * math.log10(rms) if rms > 1e-6 else None,   # dBFS real (0 = máximo digital)
            "tono": ap.clasificar_tono(f0 or 0),
            "elapsed": (time.time() - self.t0) if self.active else 0.0,
        }

    def summary(self):
        f0 = float(np.mean(self.f0s)) if self.f0s else 0.0
        vol = float(np.mean(self.vols)) if self.vols else 0.0
        return {"f0": f0, "tono": ap.clasificar_tono(f0), "rms": vol,
                "db": 20 * math.log10(vol) if vol > 1e-6 else None,
                "segundos": round(sum(len(b) for b in self.blocks) / config.SAMPLE_RATE, 1)}


class Api:
    def __init__(self):
        self._rec = _Recorder()
        self._window = None
        self._limitador = _Limitador()
        self._control = security.ControlAcceso()
        self._auth = False
        self._inicio_sesion = self._ultima_actividad = 0.0
        self._pending = None      # última grabación válida (aún sin guardar/usar)
        self._wiz = None          # estado del asistente de registro / re-entrenamiento / ampliación
        self._test_audio = None
        db.init_db()

    def _set_window(self, w):
        self._window = w

    # ---------- sesión ----------
    def _cerrar_sesion(self):
        """Bloquea la sesión y descarta cualquier dato biométrico en memoria."""
        self._auth = False
        self._pending = self._wiz = self._test_audio = None
        try:
            self._rec.stop()
        except Exception:
            pass

    def _sesion_invalida(self):
        if not self._auth:
            return "Sesión bloqueada: introduce el PIN."
        ahora = time.monotonic()
        if ahora - self._ultima_actividad > config.SESION_INACTIVIDAD_SEG and not self._rec.active:
            self._cerrar_sesion()
            return "La sesión se bloqueó por inactividad: introduce el PIN."
        if ahora - self._inicio_sesion > config.SESION_MAX_SEG:
            self._cerrar_sesion()
            return "La sesión caducó: introduce el PIN de nuevo."
        return None

    # ---------- acceso ----------
    @_safe
    def get_config(self):
        return {"tono_min": config.TONO_DURACION_MIN, "tono_max": config.TONO_DURACION_MAX,
                "pasos": config.MIN_MUESTRAS_POR_PERSONA, "max_muestras": config.MAX_MUESTRAS_POR_PERSONA}

    @_safe
    def auth_status(self):
        return {"configured": security.pin_configurado(), "authenticated": self._sesion_invalida() is None}

    @_safe
    def login(self, pin):
        if not security.pin_configurado():
            raise ApiError("No hay PIN configurado. Ejecuta: python setup_seguridad.py", "no_pin")
        if not isinstance(pin, str):
            raise ApiError("PIN inválido.", "bad_pin")
        resultado, dato = self._control.intentar(pin.strip())
        if resultado == "ok":
            # Sesión nueva: se descarta cualquier estado de una sesión anterior.
            self._cerrar_sesion()
            self._auth = True
            self._inicio_sesion = self._ultima_actividad = time.monotonic()
            return True
        if resultado == "bloqueado":
            raise ApiError(f"Demasiados intentos fallidos. Bloqueado {dato} s.", "locked")
        time.sleep(1)  # frena intentos automatizados
        raise ApiError(f"PIN incorrecto (quedan {dato} intento(s)).", "bad_pin")

    @_safe
    def logout(self):
        self._cerrar_sesion()
        return True

    # ---------- grabación (común a todas las vistas) ----------
    @_safe
    def record_start(self):
        self._pending = None
        self._rec.start()
        return True

    @_safe
    def get_live(self):
        return self._rec.live()

    @_safe
    def record_stop(self):
        audio = self._rec.stop()
        ok, motivo = ap.validar_calidad_muestra(audio)
        self._pending = audio if ok else None
        return {"valida": ok, "motivo": motivo, "segundos": round(len(audio) / config.SAMPLE_RATE, 1)}

    @_safe
    def record_cancel(self):
        self._rec.stop()
        self._pending = None
        return True

    # ---------- personas ----------
    @_safe
    def list_speakers(self):
        return [{
            "id": p["id"], "nombre": p["nombre"], "muestras": p["total_muestras"],
            "fecha": p["fecha_registro"].strftime("%Y-%m-%d %H:%M"),
            "compatibles": p["muestras_compatibles"], "reentrenar": p["necesita_reentrenar"],
        } for p in db.listar_speakers()]

    @_safe
    def rename_speaker(self, actual, nuevo):
        actual, nuevo = _nombre(actual), _nombre(nuevo)
        r = db.renombrar_speaker(actual, nuevo)
        if r == "no_existe":
            raise ApiError(f"No se encontró a '{actual}'.", "not_found")
        if r == "duplicado":
            raise ApiError(f"Ya existe una persona llamada '{nuevo}'.", "duplicate")
        return True

    @_safe
    def delete_speaker(self, nombre):
        nombre = _nombre(nombre)
        existia, archivos = db.eliminar_speaker_completo(nombre)
        if not existia:
            raise ApiError(f"No se encontró a '{nombre}'.", "not_found")
        return archivos

    # ---------- asistente de registro / re-entrenamiento / ampliación ----------
    def _wiz_info(self):
        w = self._wiz
        n = len(w["muestras"])
        lleno = n >= w["max"]
        texto = None if lleno else config.TEXTOS_LECTURA_REGISTRO[n % len(config.TEXTOS_LECTURA_REGISTRO)]
        return {"paso": n + 1, "min": w["min"], "max": w["max"], "guardadas": n,
                "completo": n >= w["min"], "lleno": lleno, "texto": texto,
                "duracion": cli._duracion_para_texto(texto) if texto else 0,
                "modo": w["modo"], "nombre": w["nombre"], "existentes": w["existentes"]}

    @_safe
    def wizard_start(self, nombre, modo):
        nombre = _nombre(nombre)
        if modo not in registro.MODOS:
            raise ApiError("Modo inválido.", "invalid")
        speaker = db.obtener_speaker(nombre)
        if modo == "registro" and speaker is not None:
            raise ApiError(f"'{nombre}' ya existe. Usa Re-entrenar o Añadir muestras en Gestionar personas.", "duplicate")
        if modo != "registro" and speaker is None:
            raise ApiError(f"No se encontró a '{nombre}'.", "not_found")
        existentes = len(speaker["muestras"]) if speaker else 0
        minimo, maximo = registro.limites_nuevas_muestras(modo, existentes)
        if maximo == 0:
            raise ApiError(f"'{nombre}' ya tiene el máximo de {config.MAX_MUESTRAS_POR_PERSONA} muestras.", "limit")
        self._wiz = {"nombre": speaker["nombre"] if speaker else nombre, "modo": modo,
                     "min": minimo, "max": maximo, "existentes": existentes, "muestras": []}
        return self._wiz_info()

    @_safe
    def wizard_keep(self):
        if not self._wiz or self._pending is None:
            raise ApiError("No hay una muestra válida para conservar.", "no_audio")
        if len(self._wiz["muestras"]) >= self._wiz["max"]:
            raise ApiError(f"Se alcanzó el máximo de {config.MAX_MUESTRAS_POR_PERSONA} muestras por persona.", "limit")
        self._wiz["muestras"].append(self._pending)
        self._pending = None
        return self._wiz_info()

    @_safe
    def wizard_finish(self):
        w = self._wiz
        if not w or len(w["muestras"]) < w["min"]:
            raise ApiError(f"Faltan muestras: el mínimo es {w['min'] if w else config.MIN_MUESTRAS_POR_PERSONA}.", "incomplete")
        n = registro.guardar_perfil(w["nombre"], w["muestras"], w["modo"])
        self._wiz = None
        return {"nombre": w["nombre"], "guardadas": n}

    @_safe
    def wizard_cancel(self):
        self._wiz, self._pending = None, None
        return True

    # ---------- identificación ----------
    @_safe
    def identify_info(self):
        todos = db.obtener_embeddings_todos()
        compat, desact = matching.filtrar_perfiles_compatibles(todos)
        if not todos:
            raise ApiError("No hay personas registradas todavía.", "empty")
        if not compat:
            raise ApiError("No hay perfiles compatibles con el motor actual: re-entrena a alguien.", "empty")
        texto = random.choice(config.TEXTOS_LECTURA_VERIFICACION)
        return {"texto": texto, "duracion": cli._duracion_para_texto(texto), "desactualizados": desact}

    @_safe
    def identify_run(self):
        audio = self._pending
        if audio is None:
            raise ApiError("No hay una grabación válida para analizar.", "no_audio")
        compat, desact = matching.filtrar_perfiles_compatibles(db.obtener_embeddings_todos())
        if not compat:
            raise ApiError("No hay perfiles compatibles con el motor actual.", "empty")
        r = matching.identificar_mejor_candidato(ap.extraer_embedding(audio), compat)
        viv = ap.calcular_puntaje_vivacidad(audio) if r.identificado else None
        self._pending = None
        return {
            "identificado": r.identificado, "nombre": r.nombre, "similitud": r.similitud,
            "distancia": r.distancia if math.isfinite(r.distancia) else None,
            "umbral_sim": config.UMBRAL_SIMILITUD,
            "vivacidad_baja": viv is not None and viv < config.VIVACIDAD_UMBRAL, "desactualizados": desact,
        }

    # ---------- análisis de voz ----------
    @_safe
    def analysis_start(self):
        self._rec.start()
        return True

    @_safe
    def analysis_stop(self):
        self._rec.stop()
        return self._rec.summary()

    # ---------- pruebas automáticas (mismas 3 del CLI, con resultado estructurado) ----------
    @_safe
    def run_test(self, n):
        if type(n) is not int or n not in (1, 2, 3):
            raise ApiError("Prueba inválida.", "invalid")
        t0 = time.time()
        if n == 1:
            try:
                audio = ap.grabar_audio(duracion=2)
            except Exception:
                log.exception("Prueba de micrófono fallida")
                raise ApiError("No se pudo grabar desde el micrófono.", "mic_unavailable")
            self._test_audio = audio
            vol = float(np.abs(audio).mean())
            estado = "ok" if vol >= 1e-6 else "warn"
            detalle = f"Nivel de señal promedio: {vol:.6f}" if estado == "ok" else "Grabó, pero parece silencio total."
        else:
            audio = self._test_audio if self._test_audio is not None else \
                (np.random.randn(config.SAMPLE_RATE * 2) * 0.01).astype(np.float32)
            if n == 2:
                if not cli.probar_base_datos(audio):
                    raise ApiError("El registro de prueba no se encontró tras guardarlo.", "db")
                estado, detalle = "ok", "Escritura, lectura y borrado correctos en MongoDB"
            else:
                score = ap.similitud_coseno(ap.extraer_embedding(audio), ap.extraer_embedding(audio))
                if score <= 0.99:
                    raise ApiError(f"Similitud inesperadamente baja: {score:.4f}", "model")
                estado, detalle = "ok", f"Similitud consigo misma: {score:.4f}"
        return {"estado": estado, "detalle": detalle, "segundos": round(time.time() - t0, 2)}

    # ---------- cierre ----------
    def _shutdown(self):
        try:
            self._cerrar_sesion()
            sd.stop()
        except Exception:
            pass

    @_safe
    def exit(self):
        self._shutdown()
        if self._window is not None:
            threading.Timer(0.2, self._window.destroy).start()
        return True
