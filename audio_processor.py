"""
audio_processor.py
-------------------
Responsable de todo lo relacionado con el audio:
 1) Capturar audio en tiempo real desde el micrófono (sounddevice).
 2) Guardarlo como archivo .wav (soundfile).
 3) Extraer un "embedding" (vector numérico que representa la voz).
 4) Comparar embeddings mediante similitud coseno.

Nota sobre el embedding usado en esta Fase 1:
    Se calculan los MFCC (Mel-Frequency Cepstral Coefficients) del audio y se
    obtiene la media y desviación estándar de cada coeficiente a lo largo del
    tiempo. Este vector es una "huella" simple pero efectiva de las
    características de una voz, no requiere descargar modelos de deep
    learning pesados y permite validar el flujo completo end-to-end
    (grabar -> guardar -> comparar) rápidamente.

    En una fase posterior, esta función se puede reemplazar por un modelo de
    embeddings pre-entrenado (por ejemplo, resemblyzer o SpeechBrain
    ECAPA-TDNN) sin tener que tocar main.py ni database.py: basta con
    modificar `extraer_embedding` manteniendo la misma firma (entra audio,
    sale un vector numpy).
"""

import datetime
import io

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

import config
import security


def precalentar_motor() -> None:
    """
    "Calienta" el motor de extracción de características antes de que el
    usuario grabe su primera muestra real. `librosa.yin` usa numba, que
    compila su código la PRIMERA vez que se llama (1-2 segundos de costo
    de arranque en frío). Si eso ocurre durante la primera grabación del
    usuario, se siente como una demora rara e inexplicable; moviéndolo
    aquí, el costo se paga una sola vez, al iniciar el programa, con un
    array diminuto y silencioso que no requiere haber grabado nada.
    """
    audio_dummy = np.zeros(int(0.5 * config.SAMPLE_RATE), dtype=np.float32)
    try:
        extraer_embedding(audio_dummy)
    except Exception:
        pass  # el precalentamiento es una optimización, nunca debe romper el arranque


def grabar_audio(duracion: int = None, samplerate: int = None) -> np.ndarray:
    """
    Graba audio desde el micrófono por defecto del sistema operativo.

    Args:
        duracion: segundos a grabar (por defecto config.DURATION).
        samplerate: frecuencia de muestreo en Hz (por defecto config.SAMPLE_RATE).

    Returns:
        np.ndarray 1D (mono) con las muestras de audio en float32.
    """
    duracion = duracion or config.DURATION
    samplerate = samplerate or config.SAMPLE_RATE

    print(f"\n🎙️  Grabando {duracion} segundos... habla ahora.")
    audio = sd.rec(
        int(duracion * samplerate),
        samplerate=samplerate,
        channels=config.CHANNELS,
        dtype="float32",
    )
    sd.wait()  # Bloquea hasta que termine la grabación
    print("✅ Grabación finalizada.")
    return audio.flatten()


def guardar_wav(audio: np.ndarray, nombre_base: str) -> str:
    """
    Guarda un array de audio como archivo .wav ENCRIPTADO dentro de
    data/audio/. El audio de voz es un dato biométrico, así que nunca se
    escribe en texto plano: se genera el .wav en memoria (soundfile sobre
    un buffer), se encripta con Fernet y solo el resultado encriptado
    toca el disco, con extensión .wav.enc.

    Args:
        audio: señal de audio (numpy array).
        nombre_base: prefijo del archivo (normalmente el nombre de la persona).

    Returns:
        Ruta (str) del archivo .wav.enc creado.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_seguro = "".join(c if c.isalnum() else "_" for c in nombre_base)
    nombre_archivo = f"{nombre_seguro}_{timestamp}.wav.enc"
    ruta = config.AUDIO_DIR / nombre_archivo

    buffer = io.BytesIO()
    sf.write(buffer, audio, config.SAMPLE_RATE, format="WAV")
    datos_encriptados = security.encriptar_bytes(buffer.getvalue())
    ruta.write_bytes(datos_encriptados)
    return str(ruta)


def _normalizar_bloque(vector: np.ndarray) -> np.ndarray:
    """
    Normaliza (L2) un bloque de características DE FORMA INDEPENDIENTE.

    Esto es clave para evitar falsos positivos: si concatenáramos MFCC
    (valores típicos ~ -100..100) con formantes en Hz (valores ~ 500..3500)
    y luego normalizáramos todo junto, los formantes dominarían la
    comparación por similitud coseno solo por tener números más grandes,
    no porque sean más relevantes. Al normalizar cada bloque por separado
    ANTES de unirlos, cada familia de características (timbre, tono,
    resonancia, energía, ritmo) aporta en igualdad de condiciones.
    """
    norma = np.linalg.norm(vector)
    return vector / norma if norma > 0 else vector


def validar_calidad_muestra(audio: np.ndarray) -> tuple:
    """
    Validación ligera de calidad de una muestra recién grabada, ANTES de
    guardarla o compararla. Solo usa estadísticas simples (media de
    amplitud y proporción de muestras saturadas) — nada de modelos
    pesados — así que es prácticamente instantánea.

    Devuelve (es_valida: bool, motivo: str). motivo queda vacío si es válida.
    """
    if audio is None or len(audio) == 0:
        return False, "No se capturó audio (posible corte del micrófono)."

    volumen = float(np.abs(audio).mean())
    if volumen < config.REGISTRO_VOLUMEN_MINIMO:
        return False, "El audio es demasiado silencioso (revisa el micrófono o habla más fuerte)."

    proporcion_saturada = float(np.mean(np.abs(audio) > 0.98))
    if proporcion_saturada > config.REGISTRO_SATURACION_MAXIMA:
        return False, "El audio tiene demasiado ruido/saturación (aléjate un poco del micrófono)."

    return True, ""


def _lpc_coeficientes(frame: np.ndarray, orden: int) -> np.ndarray:
    """
    Calcula los coeficientes LPC (Linear Predictive Coding) de un frame
    mediante autocorrelación + recursión de Levinson-Durbin — el método
    clásico y ligero (O(orden²), sin dependencias más allá de numpy) para
    modelar la envolvente espectral que produce el tracto vocal, base de
    la estimación de formantes.
    """
    autocorr = np.correlate(frame, frame, mode="full")
    r = autocorr[len(autocorr) // 2: len(autocorr) // 2 + orden + 1]

    if r[0] == 0:
        return np.zeros(orden)

    a = np.zeros(orden + 1)
    a[0] = 1.0
    error = r[0]

    for i in range(1, orden + 1):
        acumulado = r[i] - np.dot(a[1:i], r[i - 1:0:-1])
        if error == 0:
            break
        k = acumulado / error
        a_nuevo = a.copy()
        a_nuevo[1:i] = a[1:i] - k * a[i - 1:0:-1]
        a_nuevo[i] = k
        a = a_nuevo
        error *= (1 - k * k)
        if error <= 0:
            break

    return a[1:orden + 1]


def _formantes_de_frame(frame: np.ndarray, samplerate: int) -> list:
    """
    Estima los formantes de UN frame ya pre-enfatizado y enventanado:
    obtiene los coeficientes LPC, encuentra las raíces del polinomio
    resultante (cada par de raíces complejas conjugadas corresponde a una
    resonancia del tracto vocal) y convierte el ángulo de cada raíz a Hz,
    descartando polos poco confiables (fuera de rango o "demasiado anchos",
    que normalmente indican ruido y no una resonancia real).
    """
    a = _lpc_coeficientes(frame, config.LPC_ORDEN)
    coeficientes_polinomio = np.concatenate([[1.0], -a])
    raices = np.roots(coeficientes_polinomio)

    formantes = []
    for raiz in raices:
        if raiz.imag <= 0:
            continue  # una raíz por cada par conjugado alcanza
        magnitud = np.abs(raiz)
        if magnitud <= 0 or magnitud >= 1:
            continue  # polo inestable: no es una resonancia física válida

        frecuencia = np.arctan2(raiz.imag, raiz.real) * (samplerate / (2 * np.pi))
        ancho_banda = -0.5 * (samplerate / np.pi) * np.log(magnitud)

        if (config.FORMANTE_MIN_HZ <= frecuencia <= config.FORMANTE_MAX_HZ
                and ancho_banda <= config.FORMANTE_ANCHO_BANDA_MAX_HZ):
            formantes.append(frecuencia)

    formantes.sort()
    return formantes[:3]


def _analizar_formantes_y_ritmo(audio: np.ndarray, samplerate: int):
    """
    Recorre la señal UNA sola vez, en ventanas de 25ms (salto 10ms), y por
    cada ventana con voz real (RMS sobre el umbral de silencio):
      - estima sus formantes F1/F2/F3 vía LPC
      - la marca como "con voz" para después medir el tiempo de inicio
        (Voice Onset Time), la proporción de voz y las pausas internas.

    Un único recorrido (no se procesa el audio más de una vez) mantiene el
    costo bajo incluso en muestras de hasta 15 segundos.

    Devuelve (formantes_por_frame: np.ndarray Nx3, info_temporal: dict).
    """
    tam_frame = max(int(config.FRAME_FORMANTES_SEG * samplerate), config.LPC_ORDEN * 2)
    salto = max(int(config.SALTO_FORMANTES_SEG * samplerate), 1)
    ventana = np.hamming(tam_frame)

    formantes_frames = []
    voz_por_frame = []

    total_frames = max(0, (len(audio) - tam_frame) // salto + 1)
    for i in range(total_frames):
        inicio = i * salto
        frame = audio[inicio:inicio + tam_frame]

        con_voz = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) >= config.SILENCIO_UMBRAL
        voz_por_frame.append(con_voz)

        if con_voz:
            frame_ventaneado = frame * ventana
            # Pre-énfasis: resalta frecuencias altas, estándar antes de LPC
            frame_preenfasis = np.append(
                frame_ventaneado[0], frame_ventaneado[1:] - 0.97 * frame_ventaneado[:-1]
            )
            f_frame = _formantes_de_frame(frame_preenfasis, samplerate)
            if len(f_frame) == 3:
                formantes_frames.append(f_frame)

    voz_por_frame = np.array(voz_por_frame, dtype=bool)

    if len(voz_por_frame) == 0 or not voz_por_frame.any():
        info_temporal = {
            "proporcion_voz": 0.0, "tiempo_inicio_voz": 0.0,
            "num_pausas": 0, "duracion_pausa_media": 0.0,
        }
        return np.zeros((0, 3)), info_temporal

    proporcion_voz = float(voz_por_frame.mean())
    tiempo_inicio_voz = int(np.argmax(voz_por_frame)) * salto / samplerate

    # Pausas: tramos de silencio ENTRE dos tramos de voz (ignora silencio
    # inicial antes de la primera palabra y silencio final tras la última)
    num_pausas, duraciones_pausas = 0, []
    en_pausa, largo_pausa, hubo_voz = False, 0, False
    for con_voz in voz_por_frame:
        if con_voz:
            if en_pausa and hubo_voz:
                num_pausas += 1
                duraciones_pausas.append(largo_pausa * salto / samplerate)
            en_pausa, largo_pausa, hubo_voz = False, 0, True
        elif hubo_voz:
            en_pausa = True
            largo_pausa += 1

    info_temporal = {
        "proporcion_voz": proporcion_voz,
        "tiempo_inicio_voz": tiempo_inicio_voz,
        "num_pausas": num_pausas,
        "duracion_pausa_media": float(np.mean(duraciones_pausas)) if duraciones_pausas else 0.0,
    }

    resultado_formantes = np.array(formantes_frames) if formantes_frames else np.zeros((0, 3))
    return resultado_formantes, info_temporal


def extraer_embedding(audio: np.ndarray, samplerate: int = None) -> np.ndarray:
    """
    Motor biométrico avanzado: construye un embedding multi-factorial
    combinando 5 familias de características acústicas independientes,
    cada una normalizada (L2) por separado antes de unirse:

      1) MFCC (media+std)        -> timbre general del tracto vocal
      2) F0 (media+std)          -> tono / entonación
      3) Formantes F1,F2,F3      -> resonancia específica del tracto vocal
         (media+std, vía LPC/Levinson-Durbin)
      4) Energía espectral       -> RMS, centroide, ancho de banda y
         (media+std)                planitud espectral (intensidad y "perfil de ruido")
      5) Patrones temporales     -> proporción de voz, tiempo de inicio
                                     (VOT), número y duración media de pausas (ritmo)

    Args:
        audio: señal de audio (numpy array, mono).
        samplerate: frecuencia de muestreo (por defecto config.SAMPLE_RATE).

    Returns:
        np.ndarray 1D con el embedding combinado (60 dimensiones con la
        configuración por defecto: 40 MFCC + 2 F0 + 6 formantes + 4
        temporales + 8 energía).
    """
    samplerate = samplerate or config.SAMPLE_RATE

    if audio is None or len(audio) == 0:
        raise ValueError("El audio recibido está vacío; no se puede extraer un embedding.")

    audio = audio.astype(np.float32)

    # --- 1) MFCC: timbre general ---
    mfcc = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=config.N_MFCC)
    bloque_mfcc = _normalizar_bloque(
        np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float64)
    )
    del mfcc  # libera el espectrograma MFCC apenas se resume a media/std

    # --- 2) F0: tono ---
    f0_serie = librosa.yin(audio, fmin=config.F0_MIN, fmax=config.F0_MAX, sr=samplerate)
    f0_validos = f0_serie[(f0_serie > config.F0_MIN) & (f0_serie < config.F0_MAX)]
    f0_media = float(np.mean(f0_validos)) if len(f0_validos) else 0.0
    f0_std = float(np.std(f0_validos)) if len(f0_validos) else 0.0
    bloque_f0 = _normalizar_bloque(np.array([f0_media, f0_std], dtype=np.float64))
    del f0_serie

    # --- 3) Formantes + patrones temporales (un solo recorrido de la señal) ---
    formantes_frames, info_temporal = _analizar_formantes_y_ritmo(audio, samplerate)
    if len(formantes_frames) > 0:
        formantes_media = formantes_frames.mean(axis=0)
        formantes_std = formantes_frames.std(axis=0)
    else:
        formantes_media = np.zeros(3)
        formantes_std = np.zeros(3)
    bloque_formantes = _normalizar_bloque(
        np.concatenate([formantes_media, formantes_std]).astype(np.float64)
    )
    del formantes_frames

    bloque_temporal = _normalizar_bloque(np.array([
        info_temporal["proporcion_voz"],
        info_temporal["tiempo_inicio_voz"],
        float(info_temporal["num_pausas"]),
        info_temporal["duracion_pausa_media"],
    ], dtype=np.float64))

    # --- 4) Energía / distribución espectral / perfil de ruido ---
    rms = librosa.feature.rms(y=audio)[0]
    centroide = librosa.feature.spectral_centroid(y=audio, sr=samplerate)[0]
    ancho_banda = librosa.feature.spectral_bandwidth(y=audio, sr=samplerate)[0]
    planitud = librosa.feature.spectral_flatness(y=audio)[0]
    bloque_energia = _normalizar_bloque(np.array([
        rms.mean(), rms.std(),
        centroide.mean(), centroide.std(),
        ancho_banda.mean(), ancho_banda.std(),
        planitud.mean(), planitud.std(),
    ], dtype=np.float64))
    del rms, centroide, ancho_banda, planitud  # libera los arreglos espectrales de inmediato

    embedding = np.concatenate([
        bloque_mfcc, bloque_f0, bloque_formantes, bloque_temporal, bloque_energia
    ])

    return embedding


def cargar_audio_desde_archivo(ruta: str) -> np.ndarray:
    """
    Carga un archivo .wav.enc ya guardado en disco: lo desencripta en
    memoria y lo decodifica con librosa. Útil para pruebas/depuración.
    """
    datos_encriptados = open(ruta, "rb").read()
    datos_planos = security.desencriptar_bytes(datos_encriptados)
    audio, _ = librosa.load(io.BytesIO(datos_planos), sr=config.SAMPLE_RATE, mono=True)
    return audio


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos embeddings (valor entre -1 y 1;
    en la práctica, entre 0 y 1 para voces reales). 1.0 = idénticos.
    """
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norma_a * norma_b))


# ---------------------------------------------------------------------------
# ANÁLISIS DE MODULACIÓN DE VOZ EN TIEMPO REAL (F0 - frecuencia fundamental)
# ---------------------------------------------------------------------------
# Estas funciones son independientes del flujo de enrollment/identificación:
# no modifican los embeddings MFCC ni la lógica de comparación ya existente.
# Se usan para estimar, mientras la persona habla, si su tono tiende a ser
# más GRAVE o más AGUDO, y para dibujar esa evolución en la terminal.

def clasificar_tono(f0_hz: float) -> str:
    """
    Clasifica una frecuencia fundamental (Hz) como GRAVE, MEDIO o AGUDO
    según los umbrales definidos en config.py.
    """
    if f0_hz is None or f0_hz <= 0:
        return "Sin datos"
    if f0_hz < config.F0_UMBRAL_GRAVE:
        return "GRAVE"
    elif f0_hz < config.F0_UMBRAL_AGUDO:
        return "MEDIO"
    else:
        return "AGUDO"


def _estimar_f0_bloque(bloque: np.ndarray, samplerate: int):
    """
    Estima la frecuencia fundamental (F0) de un pequeño bloque de audio
    usando el algoritmo YIN (librosa.yin), que es rápido y adecuado para
    análisis casi en tiempo real. Devuelve None si el bloque es silencio
    o si no se detecta un tono confiable.
    """
    if np.abs(bloque).mean() < config.SILENCIO_UMBRAL:
        return None  # silencio: no tiene sentido estimar tono

    try:
        f0_serie = librosa.yin(
            bloque.astype(np.float32),
            fmin=config.F0_MIN,
            fmax=config.F0_MAX,
            sr=samplerate,
        )
        f0_validos = f0_serie[(f0_serie > config.F0_MIN) & (f0_serie < config.F0_MAX)]
        if len(f0_validos) == 0:
            return None
        return float(np.median(f0_validos))
    except Exception:
        # Si el bloque es demasiado corto o ruidoso, simplemente se omite
        return None


def _calcular_volumen_rms(bloque: np.ndarray) -> float:
    """
    Calcula el nivel de volumen de un bloque de audio como RMS
    (Root Mean Square), la forma estándar de medir la "energía" o
    intensidad percibida de una señal de audio. Es una operación muy
    liviana (un cuadrado, una media y una raíz), ideal para actualizarse
    en tiempo real sin consumir CPU de más.
    """
    if bloque is None or len(bloque) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(bloque, dtype=np.float64))))


def _mostrar_barra_en_vivo(f0, volumen: float) -> None:
    """
    Dibuja en la terminal, sobre la misma línea (usando \\r), DOS barras
    ASCII en vivo:
      - Tono (grave/agudo): el marcador ● se desplaza según la F0 detectada.
      - Volumen: una barra de nivel que se llena según la intensidad (RMS)
        de la voz, igual que un medidor de sonido de un micrófono real.
    """
    ancho = config.ANCHO_BARRA_VIVO

    # --- Barra de tono ---
    if f0 is None:
        barra_tono = "─" * ancho
        etiqueta = "silencio"
        valor_txt = "-----Hz"
    else:
        f0_acotado = max(config.F0_VIS_MIN, min(config.F0_VIS_MAX, f0))
        proporcion = (f0_acotado - config.F0_VIS_MIN) / (config.F0_VIS_MAX - config.F0_VIS_MIN)
        posicion = int(proporcion * (ancho - 1))
        barra_tono = "".join("●" if i == posicion else "─" for i in range(ancho))
        etiqueta = clasificar_tono(f0)
        valor_txt = f"{f0:5.0f}Hz"

    # --- Barra de volumen ---
    nivel = max(0.0, min(1.0, volumen / config.VOLUMEN_VIS_MAX))
    llenas = int(nivel * ancho)
    barra_volumen = "█" * llenas + "░" * (ancho - llenas)

    linea = (
        f"\r🎵 TONO [{barra_tono}] {valor_txt} {etiqueta:8s} "
        f"│ 🔊 VOL [{barra_volumen}] {nivel * 100:3.0f}%"
    )
    print(linea, end="", flush=True)


def grabar_audio_con_analisis(duracion: int = None, samplerate: int = None, guardar_audio: bool = False):
    """
    Graba audio desde el micrófono igual que `grabar_audio`, pero además
    analiza EN VIVO, mientras se graba:
      - el tono (F0): grave / medio / agudo
      - el volumen (RMS): qué tan fuerte se está hablando
    y muestra ambos como barras ASCII que se actualizan en tiempo real.

    Optimización para entornos livianos (nube/contenedores):
        Por defecto (`guardar_audio=False`) esta función NO acumula el
        audio completo en memoria ni lo copia bloque a bloque, porque el
        análisis en vivo no necesita conservar la señal después de medirla.
        Esto reduce el uso de RAM y evita el costo de CPU de concatenar
        arreglos grandes al final. Si en el futuro necesitas guardar esta
        grabación (por ejemplo, para usarla también como muestra de
        enrollment), pasa `guardar_audio=True`.

    Returns:
        tuple(audio_completo, f0_promedio, clasificacion, volumen_promedio)
            audio_completo: np.ndarray (vacío si guardar_audio=False)
            f0_promedio: promedio de F0 detectado durante la grabación (Hz)
            clasificacion: "GRAVE", "MEDIO", "AGUDO" o "Sin datos"
            volumen_promedio: RMS promedio detectado durante la grabación
    """
    duracion = duracion or config.DURATION
    samplerate = samplerate or config.SAMPLE_RATE
    bloque_muestras = int(config.BLOQUE_ANALISIS_SEG * samplerate)

    audio_bloques = [] if guardar_audio else None
    f0_valores = []
    volumen_valores = []

    print(f"\n🎙️  Grabando {duracion}s con análisis en vivo (tono + volumen). Habla ahora.\n")

    def _callback(indata, frames, time_info, status):
        # indata es un buffer reutilizado por sounddevice: solo se copia si
        # realmente vamos a conservarlo (guardar_audio=True). Si no, se lee
        # directamente aquí mismo, lo cual es más liviano en CPU/RAM.
        bloque = indata[:, 0]

        volumen = _calcular_volumen_rms(bloque)
        volumen_valores.append(volumen)

        f0 = _estimar_f0_bloque(bloque, samplerate)
        if f0 is not None:
            f0_valores.append(f0)

        if guardar_audio:
            audio_bloques.append(bloque.copy())

        _mostrar_barra_en_vivo(f0, volumen)

    with sd.InputStream(
        samplerate=samplerate,
        channels=config.CHANNELS,
        dtype="float32",
        blocksize=bloque_muestras,
        callback=_callback,
    ):
        sd.sleep(int(duracion * 1000))

    print()  # salto de línea al terminar, para no pisar la última barra
    print("✅ Grabación y análisis finalizados.")

    if guardar_audio and audio_bloques:
        audio_completo = np.concatenate(audio_bloques)
    else:
        audio_completo = np.array([], dtype=np.float32)

    f0_promedio = float(np.mean(f0_valores)) if f0_valores else 0.0
    clasificacion = clasificar_tono(f0_promedio)
    volumen_promedio = float(np.mean(volumen_valores)) if volumen_valores else 0.0

    return audio_completo, f0_promedio, clasificacion, volumen_promedio