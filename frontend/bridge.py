"""
bridge.py — capa de comunicación Frontend ⇄ Backend (expuesta a JS vía pywebview).

Mapa UI → backend existente (no se duplica lógica, solo se orquesta):
  Registrar / Re-entrenar : ap.validar_calidad_muestra, ap.guardar_wav, ap.extraer_embedding,
                            db.obtener_o_crear_speaker, db.guardar_muestra, db.eliminar_muestras_speaker
  Identificar             : matching.filtrar_perfiles_compatibles / identificar_mejor_candidato,
                            ap.extraer_embedding, ap.calcular_puntaje_vivacidad
  Eliminar                : db.eliminar_speaker_completo
  Gestionar               : db.listar_speakers, db.diagnostico_embeddings, db.renombrar_speaker
  Análisis de voz         : ap._estimar_f0_bloque, ap._calcular_volumen_rms, ap.clasificar_tono
  Pruebas                 : ap.grabar_audio, db.*, ap.extraer_embedding, ap.similitud_coseno
  PIN                     : security.verificar_pin (mismo PIN/.env que el CLI)

Limitación: el backend graba con duración fija y bloqueante (grabar_audio*), incompatible con
Iniciar/Detener y niveles en vivo. Por eso _Recorder abre su propio sd.InputStream, pero reutiliza
las funciones de análisis del backend. Todo método devuelve {ok, data, error, code}.
"""
import functools
import math
import random
import threading
import time
from contextlib import closing

import numpy as np
import sounddevice as sd

import audio_processor as ap
import config
import database as db
import main as cli  # solo se reutiliza _duracion_para_texto; importar main no ejecuta nada
import matching
import security

_BLOQUE = int(0.1 * config.SAMPLE_RATE)                              # 100 ms: nivel/onda fluidos
_VENTANA_F0 = int(config.BLOQUE_ANALISIS_SEG * config.SAMPLE_RATE)   # 300 ms, igual que el CLI
_ABIERTOS = {"auth_status", "login", "get_config"}


class ApiError(Exception):
    def __init__(self, msg, code="error"):
        super().__init__(msg)
        self.code = code


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        if fn.__name__ not in _ABIERTOS and not self._auth:
            return {"ok": False, "data": None, "error": "Sesión bloqueada: introduce el PIN.", "code": "auth"}
        try:
            return {"ok": True, "data": fn(self, *a, **k), "error": None, "code": None}
        except ApiError as e:
            return {"ok": False, "data": None, "error": str(e), "code": e.code}
        except Exception as e:  # nunca dejar que una excepción rompa el puente
            return {"ok": False, "data": None, "error": f"{type(e).__name__}: {e}", "code": "backend_error"}
    return wrapper


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
        self.ring = np.zeros(0, dtype=np.float32)

    def procesar(self, b):
        rms = ap._calcular_volumen_rms(b)
        with self._lock:
            self.blocks.append(np.array(b, dtype=np.float32))
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
        if self.active:
            raise ApiError("Ya hay una grabación en curso.", "busy")
        with self._lock:
            self._reset()
        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, dtype="float32",
                blocksize=_BLOQUE, callback=lambda indata, f, t, s: self.procesar(indata[:, 0]),
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise ApiError(f"Micrófono no disponible: {e}", "mic_unavailable")
        self.active, self.t0 = True, time.time()

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
        self._auth = False
        self._fails, self._lock_until = 0, 0.0
        self._pending = None      # última grabación válida (aún sin guardar/usar)
        self._wiz = None          # estado del asistente de registro / re-entrenamiento
        self._test_audio = None
        db.init_db()

    def set_window(self, w):
        self._window = w

    # ---------- acceso ----------
    @_safe
    def get_config(self):
        return {"tono_min": config.TONO_DURACION_MIN, "tono_max": config.TONO_DURACION_MAX,
                "pasos": config.NUM_PASOS_REGISTRO}

    @_safe
    def auth_status(self):
        return {"configured": bool(config.PIN_SALT and config.PIN_HASH), "authenticated": self._auth}

    @_safe
    def login(self, pin):
        if not (config.PIN_SALT and config.PIN_HASH):
            raise ApiError("No hay PIN configurado. Ejecuta: python setup_seguridad.py", "no_pin")
        if time.time() < self._lock_until:
            raise ApiError(f"Bloqueado. Espera {int(self._lock_until - time.time()) + 1} s.", "locked")
        if security.verificar_pin(str(pin).strip(), config.PIN_SALT, config.PIN_HASH):
            self._auth, self._fails = True, 0
            return True
        self._fails += 1
        if self._fails >= security.MAX_INTENTOS:
            self._fails, self._lock_until = 0, time.time() + security.BLOQUEO_SEGUNDOS
            raise ApiError(f"Demasiados intentos. Bloqueado {security.BLOQUEO_SEGUNDOS} s.", "locked")
        raise ApiError(f"PIN incorrecto ({self._fails}/{security.MAX_INTENTOS})", "bad_pin")

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
        with closing(db.SessionLocal()) as s:
            diag = {d["nombre"]: d for d in db.diagnostico_embeddings(s)}
            return [{
                "id": sp.id, "nombre": sp.nombre, "muestras": len(sp.muestras),
                "fecha": sp.fecha_registro.strftime("%Y-%m-%d %H:%M"),
                "compatibles": diag[sp.nombre]["muestras_compatibles"],
                "reentrenar": diag[sp.nombre]["necesita_reentrenar"],
            } for sp in db.listar_speakers(s)]

    @_safe
    def rename_speaker(self, actual, nuevo):
        nuevo = (nuevo or "").strip()
        if not nuevo:
            raise ApiError("El nombre nuevo no puede estar vacío.", "invalid")
        with closing(db.SessionLocal()) as s:
            r = db.renombrar_speaker(s, actual, nuevo)
        if r == "no_existe":
            raise ApiError(f"No se encontró a '{actual}'.", "not_found")
        if r == "duplicado":
            raise ApiError(f"Ya existe una persona llamada '{nuevo}'.", "duplicate")
        return True

    @_safe
    def delete_speaker(self, nombre):
        existia, archivos = db.eliminar_speaker_completo(nombre)
        if not existia:
            raise ApiError(f"No se encontró a '{nombre}'.", "not_found")
        return archivos

    # ---------- asistente de registro / re-entrenamiento ----------
    def _wiz_info(self):
        w, total = self._wiz, config.NUM_PASOS_REGISTRO
        n = len(w["muestras"])
        texto = None if n >= total else config.TEXTOS_LECTURA_REGISTRO[n % len(config.TEXTOS_LECTURA_REGISTRO)]
        return {"paso": n + 1, "total": total, "guardadas": n, "completo": n >= total, "texto": texto,
                "duracion": cli._duracion_para_texto(texto) if texto else 0, "modo": w["modo"], "nombre": w["nombre"]}

    @_safe
    def wizard_start(self, nombre, modo):
        nombre = (nombre or "").strip()
        if not nombre:
            raise ApiError("El nombre no puede estar vacío.", "invalid")
        with closing(db.SessionLocal()) as s:
            existe = s.query(db.Speaker).filter_by(nombre=nombre).first() is not None
        if modo == "registro" and existe:
            raise ApiError(f"'{nombre}' ya existe. Usa Re-entrenar en Gestionar personas.", "duplicate")
        if modo == "reentrenar" and not existe:
            raise ApiError(f"No se encontró a '{nombre}'.", "not_found")
        self._wiz = {"nombre": nombre, "modo": modo, "muestras": []}
        return self._wiz_info()

    @_safe
    def wizard_keep(self):
        if not self._wiz or self._pending is None:
            raise ApiError("No hay una muestra válida para conservar.", "no_audio")
        self._wiz["muestras"].append(self._pending)
        self._pending = None
        return self._wiz_info()

    @_safe
    def wizard_finish(self):
        w = self._wiz
        if not w or len(w["muestras"]) < config.NUM_PASOS_REGISTRO:
            raise ApiError("Faltan muestras para completar.", "incomplete")
        with closing(db.SessionLocal()) as s:
            if w["modo"] == "reentrenar":
                sp = s.query(db.Speaker).filter_by(nombre=w["nombre"]).first()
                if sp is None:
                    raise ApiError(f"No se encontró a '{w['nombre']}'.", "not_found")
                db.eliminar_muestras_speaker(s, sp)   # solo ahora: las nuevas ya están completas
            else:
                sp = db.obtener_o_crear_speaker(s, w["nombre"])
            for audio in w["muestras"]:
                ruta = ap.guardar_wav(audio, nombre_base=w["nombre"])
                db.guardar_muestra(s, sp, ruta, ap.extraer_embedding(audio))
        nombre, self._wiz = w["nombre"], None
        return nombre

    @_safe
    def wizard_cancel(self):
        self._wiz, self._pending = None, None
        return True

    # ---------- identificación ----------
    @_safe
    def identify_info(self):
        with closing(db.SessionLocal()) as s:
            todos = db.obtener_embeddings_todos(s)
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
        with closing(db.SessionLocal()) as s:
            todos = db.obtener_embeddings_todos(s)
        compat, desact = matching.filtrar_perfiles_compatibles(todos)
        if not compat:
            raise ApiError("No hay perfiles compatibles con el motor actual.", "empty")
        r = matching.identificar_mejor_candidato(ap.extraer_embedding(audio), compat)
        viv = ap.calcular_puntaje_vivacidad(audio) if r.identificado else None
        self._pending = None
        return {
            "identificado": r.identificado, "nombre": r.nombre, "similitud": r.similitud,
            "distancia": r.distancia if math.isfinite(r.distancia) else None,
            "umbral_sim": config.UMBRAL_SIMILITUD, "umbral_dist": config.UMBRAL_DISTANCIA_EUCLIDIANA,
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
        t0 = time.time()
        if n == 1:
            try:
                audio = ap.grabar_audio(duracion=2)
            except Exception as e:
                raise ApiError(f"No se pudo grabar: {e}", "mic_unavailable")
            self._test_audio = audio
            vol = float(np.abs(audio).mean())
            estado = "ok" if vol >= 1e-6 else "warn"
            detalle = f"Nivel de señal promedio: {vol:.6f}" if estado == "ok" else "Grabó, pero parece silencio total."
        else:
            audio = self._test_audio if self._test_audio is not None else \
                (np.random.randn(config.SAMPLE_RATE * 2) * 0.01).astype(np.float32)
            if n == 2:
                nombre = "_usuario_prueba_"
                with closing(db.SessionLocal()) as s:
                    sp = db.obtener_o_crear_speaker(s, nombre)
                    db.guardar_muestra(s, sp, ap.guardar_wav(audio, "_prueba_"), ap.extraer_embedding(audio))
                    ok = any(x == nombre for x, _, _ in db.obtener_embeddings_todos(s))
                db.eliminar_speaker_completo(nombre)  # limpieza: el registro de prueba no queda en la BD
                if not ok:
                    raise ApiError("El registro de prueba no se encontró tras guardarlo.", "db")
                estado, detalle = "ok", f"Escritura/lectura correctas en {config.DB_PATH.name}"
            else:
                score = ap.similitud_coseno(ap.extraer_embedding(audio), ap.extraer_embedding(audio))
                if score <= 0.99:
                    raise ApiError(f"Similitud inesperadamente baja: {score:.4f}", "model")
                estado, detalle = "ok", f"Similitud consigo misma: {score:.4f}"
        return {"estado": estado, "detalle": detalle, "segundos": round(time.time() - t0, 2)}

    # ---------- cierre ----------
    def shutdown(self):
        try:
            self._rec.stop()
            sd.stop()
        except Exception:
            pass

    @_safe
    def exit(self):
        self.shutdown()
        if self._window is not None:
            threading.Timer(0.2, self._window.destroy).start()
        return True
