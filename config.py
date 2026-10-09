"""
config.py
---------
Configuración global del sistema de identificación de voz (Speaker ID).
Centraliza rutas y parámetros para que el resto de módulos no tengan
valores "quemados" (hardcoded) y sea fácil ajustar el comportamiento.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carga las variables definidas en .env (si el archivo existe) al entorno
# del proceso. Debe ejecutarse ANTES de leer cualquier os.getenv() de abajo.
load_dotenv()

# ---------------------------------------------------------------------------
# Seguridad: PIN de acceso y clave de encriptación (leídos desde .env)
# ---------------------------------------------------------------------------
# Estos valores NUNCA se escriben aquí a mano — se generan una vez con
# `python setup_seguridad.py`, que los guarda en .env (archivo que está en
# .gitignore y nunca se sube al repositorio). Si no existen, security.py
# lo detecta y pide correr el script de configuración.
PIN_SALT = os.getenv("VOICE_ID_PIN_SALT")
PIN_HASH = os.getenv("VOICE_ID_PIN_HASH")
FERNET_KEY = os.getenv("VOICE_ID_FERNET_KEY")
# Iteraciones PBKDF2 con las que se generó el hash del PIN. Los .env creados
# antes de este cambio no traen la variable y usaban 200 000; los nuevos
# (setup_seguridad.py) usan 600 000, el mínimo recomendado por OWASP (2023).
PIN_ITERACIONES = int(os.getenv("VOICE_ID_PIN_ITER", "200000"))

# ---------------------------------------------------------------------------
# Base de datos: MongoDB (no relacional)
# ---------------------------------------------------------------------------
# La cadena de conexión (que puede incluir usuario y contraseña) se lee
# SIEMPRE del .env — nunca se escribe en el código. Por defecto apunta a un
# MongoDB local sin credenciales. Para un servidor remoto (p. ej. Atlas) usa
# "mongodb+srv://usuario:clave@cluster/..." o añade "tls=true": la conexión
# a un host remoto sin TLS se rechaza (ver database._validar_uri).
MONGODB_URI = os.getenv("VOICE_ID_MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DB = os.getenv("VOICE_ID_MONGODB_DB", "voice_id")
MONGODB_TIMEOUT_MS = int(os.getenv("VOICE_ID_MONGODB_TIMEOUT_MS", "5000"))

# ---------------------------------------------------------------------------
# Optimización para entornos de nube / contenedores livianos
# ---------------------------------------------------------------------------
# librosa usa numba y numpy por debajo, que por defecto intentan usar TODOS
# los núcleos disponibles para paralelizar. En un contenedor con CPU limitada
# (por ejemplo 1 vCPU en un servidor económico) esto genera más hilos de los
# que la máquina puede atender realmente, lo cual consume memoria extra sin
# aportar velocidad. Fijar estas variables a 1 hilo reduce el uso de RAM y
# hace el consumo de CPU predecible. Se definen ANTES de importar librosa
# (en audio_processor.py) para que tengan efecto.
# Se puede sobreescribir desde fuera (por ejemplo en el Dockerfile o en la
# configuración del servidor) si la máquina tiene más núcleos disponibles.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Rutas base del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"          # Muestras de audio (.wav) y embeddings (.npy)
LOG_DIR = DATA_DIR / "logs"            # Registro técnico de errores (sin datos biométricos)
SQLITE_LEGADO_PATH = DATA_DIR / "voice_id.db"  # BD SQLite de versiones anteriores (solo para migrar)

# Crear carpetas necesarias si no existen (se ejecuta al importar el módulo).
# mode=0o700: solo el usuario dueño puede leerlas (en Windows se heredan los
# permisos de la carpeta del usuario, que ya son privados).
for _carpeta in (DATA_DIR, AUDIO_DIR, LOG_DIR):
    _carpeta.mkdir(mode=0o700, parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parámetros de audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000     # Hz — frecuencia estándar para procesamiento de voz
DURATION = 4            # segundos por grabación
CHANNELS = 1            # mono

# ---------------------------------------------------------------------------
# Parámetros de enrollment (registro de personas)
# ---------------------------------------------------------------------------
# Cada persona debe tener entre MIN y MAX muestras de voz guardadas. El
# mínimo garantiza una huella robusta; el máximo acota el tamaño del
# documento en MongoDB y el coste de cada identificación. Se hace cumplir
# en tres capas: la interfaz, database.py y el validador $jsonSchema de la
# colección (ver database.init_db).
MIN_MUESTRAS_POR_PERSONA = 4
MAX_MUESTRAS_POR_PERSONA = 25

# ---------------------------------------------------------------------------
# Parámetros de identificación
# ---------------------------------------------------------------------------
UMBRAL_SIMILITUD = 0.995  # umbral de similitud coseno (0-1) para aceptar una coincidencia
                           # Todos los embeddings tienen la misma norma (9 bloques L2 -> norma 3),
                           # así que distancia² = 18·(1 - coseno): 0.995 equivale a distancia <= 0.30.
                           # Calibrado con prueba sintética (4 "personas", 3 muestras c/u):
                           # misma persona >= 0.9983, personas distintas <= 0.9949. Un umbral más
                           # bajo (p. ej. 0.90) acepta a personas distintas: ajústalo solo con
                           # pruebas de voces reales, nunca bajándolo "por si acaso".

# ---------------------------------------------------------------------------
# Versión del pipeline de extracción de embeddings
# ---------------------------------------------------------------------------
# Sube este número cada vez que cambies QUÉ características entran al
# embedding (agregar/quitar bloques, cambiar N_MFCC, etc.). Cada embedding
# se "sella" con esta versión al guardarse (ver database.guardar_muestra).
# Si luego mejoras el pipeline y la dimensión cambia, los perfiles viejos
# se detectan automáticamente como desactualizados — en vez de romper el
# programa comparando vectores de tamaños distintos — y el sistema avisa
# qué personas necesitan re-entrenarse (Gestionar personas → Re-entrenar).
EMBEDDING_VERSION = 2
EMBEDDING_DIM = 156   # dimensión esperada del embedding con la versión actual

# ---------------------------------------------------------------------------
# Parámetros de extracción de características (MFCC)
# ---------------------------------------------------------------------------
N_MFCC = 20   # número de coeficientes MFCC a extraer por frame

# ---------------------------------------------------------------------------
# Parámetros de análisis de tono / modulación de voz (F0 - frecuencia fundamental)
# ---------------------------------------------------------------------------
BLOQUE_ANALISIS_SEG = 0.3     # segundos de audio analizados en cada actualización en vivo
SILENCIO_UMBRAL = 0.01        # amplitud promedio mínima para considerar que hay voz (no silencio)

# Rango de duración PERMITIDO para la Opción 4 (Analizar modulación de voz).
# Por debajo de 7s el promedio de F0 puede ser poco confiable; por encima de
# 15s no aporta precisión adicional y solo alarga la espera del usuario.
TONO_DURACION_MIN = 7
TONO_DURACION_MAX = 15

F0_MIN = 50.0                 # Hz — frecuencia mínima de búsqueda (voz humana grave)
F0_MAX = 400.0                # Hz — frecuencia máxima de búsqueda (voz humana aguda)

F0_UMBRAL_GRAVE = 140.0       # por debajo de este valor, la voz se clasifica como GRAVE
F0_UMBRAL_AGUDO = 220.0       # por encima de este valor, la voz se clasifica como AGUDA
                               # (entre ambos umbrales se clasifica como MEDIA)

F0_VIS_MIN = 60.0             # rango usado únicamente para dibujar la barra de tono en terminal
F0_VIS_MAX = 300.0

# ---------------------------------------------------------------------------
# Parámetros de la barra de nivel de sonido (volumen) en tiempo real
# ---------------------------------------------------------------------------
VOLUMEN_VIS_MAX = 0.30        # amplitud RMS considerada "100%" para la barra de volumen
ANCHO_BARRA_VIVO = 28         # ancho (en caracteres) de cada barra ASCII en pantalla

# ---------------------------------------------------------------------------
# Motor biométrico avanzado: formantes (LPC) y patrones temporales/ritmo
# ---------------------------------------------------------------------------
# Los formantes (F1, F2, F3) son las frecuencias de resonancia del tracto
# vocal: junto al MFCC (timbre) y al F0 (tono), forman una huella mucho
# más específica de cada persona que el tono por sí solo.
LPC_ORDEN = 16                     # orden del modelo LPC (regla práctica: ~2 + muestreo/1000)
FRAME_FORMANTES_SEG = 0.025        # ventana de análisis: 25 ms (estándar en fonética)
SALTO_FORMANTES_SEG = 0.010        # salto entre ventanas: 10 ms
FORMANTE_MIN_HZ = 90.0             # rango de frecuencias donde pueden existir F1/F2/F3
FORMANTE_MAX_HZ = 5000.0
FORMANTE_ANCHO_BANDA_MAX_HZ = 400.0  # descarta polos LPC "anchos" (ruido, no formante real)

# ---------------------------------------------------------------------------
# Validación de calidad de una muestra recién grabada (evita falsos
# positivos por audio silencioso o saturado antes de guardarla/compararla)
# ---------------------------------------------------------------------------
REGISTRO_VOLUMEN_MINIMO = 0.01        # RMS mínimo para considerar que hay voz real
REGISTRO_SATURACION_MAXIMA = 0.05     # máx. proporción de muestras "clippeadas" (ruido excesivo)

# ---------------------------------------------------------------------------
# Sistema de textos de lectura guiada
# ---------------------------------------------------------------------------
# El registro siempre son estos 4 pasos secuenciales, cada uno con un texto
# distinto (fonéticamente variado: incluye la mayoría de vocales/consonantes
# del español) para construir una huella robusta. La verificación usa un
# texto de un banco COMPLETAMENTE separado, para no depender de la frase
# exacta memorizada durante el registro.
NUM_PASOS_REGISTRO = MIN_MUESTRAS_POR_PERSONA   # pasos obligatorios; luego son opcionales hasta el máximo

TEXTOS_LECTURA_REGISTRO = [
    "El veloz murciélago hindú comía feliz cardillo y kiwi mientras el sol se ocultaba.",
    "Un jugoso zapallo brilla bajo la luz del mediodía en el jardín de la abuela.",
    "Quiere la boca exhausta vid, kiwi, piña y fugaz zumo bajo el árbol del patio.",
    "Benjamín pidió una bebida de kiwi y fresa junto al río mientras cantaba una canción.",
]

TEXTOS_LECTURA_VERIFICACION = [
    "Hoy quiero comprobar quién soy hablando con calma y claridad frente al micrófono.",
    "Esta frase es distinta a las que usé para registrarme, y así lo demuestro ahora.",
    "Mi identidad se confirma por cómo hablo, no por las palabras exactas que digo hoy.",
]

# ---------------------------------------------------------------------------
# Preprocesamiento: VAD (recorte de silencio) y estimación de SNR
# ---------------------------------------------------------------------------
# VAD_TOP_DB: umbral (en dB por debajo del pico) que usa librosa.effects.trim
# para decidir qué es "silencio" y recortarlo de los extremos antes de
# extraer características — así el embedding no se "ensucia" con silencio
# de sobra al inicio/final de la grabación.
VAD_TOP_DB = 30.0

# SNR_MINIMO_DB: relación señal-ruido mínima aceptable. Por debajo de esto
# se rechaza la muestra (demasiado ruido de fondo/eco para confiar en ella).
SNR_MINIMO_DB = 8.0

# La identificación se decide solo con UMBRAL_SIMILITUD (ver arriba). La
# distancia euclidiana se muestra como dato informativo: con embeddings de
# norma fija aporta la misma información que la similitud coseno.

# ---------------------------------------------------------------------------
# Heurística de "vivacidad" (liveness) — ADVERTENCIA, no bloqueo automático
# ---------------------------------------------------------------------------
# Ver el docstring de audio_processor.calcular_puntaje_vivacidad: esto es
# una señal débil e informativa, NO un sistema de anti-spoofing robusto.
VIVACIDAD_UMBRAL = 0.15

# ---------------------------------------------------------------------------
# Cuenta regresiva antes de grabar
# ---------------------------------------------------------------------------
CUENTA_REGRESIVA_SEG = 3

# ---------------------------------------------------------------------------
# Duración de grabación adaptada a la longitud del texto a leer
# ---------------------------------------------------------------------------
REGISTRO_DURACION_MIN = 10
REGISTRO_DURACION_MAX = 15
PALABRAS_POR_SEGUNDO_LECTURA = 2.0   # velocidad de lectura conservadora (~120 palabras/min)
MARGEN_LECTURA_SEG = 3               # colchón extra para prepararse/terminar sin prisa

# ---------------------------------------------------------------------------
# Sesión de la interfaz gráfica y límites anti-abuso (DoS)
# ---------------------------------------------------------------------------
SESION_INACTIVIDAD_SEG = 10 * 60     # se bloquea tras 10 min sin actividad
SESION_MAX_SEG = 8 * 60 * 60         # y siempre tras 8 h, aunque haya actividad
GRABACION_MAX_SEG = 30               # tope duro de una grabación (memoria acotada)
NOMBRE_MAX_LARGO = 60                # longitud máxima del nombre de una persona
