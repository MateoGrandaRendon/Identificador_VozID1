# Sistema de Identificación de Voz — Fase 1

Identificación de personas por su voz (Speaker Identification), no
reconocimiento de lo que dicen (eso sería speech-to-text).

## 1. Estructura del proyecto

```
voice_identifier/
├── main.py                  # Menú CLI: registro, identificación, gestión, pruebas
├── frontend/                # Interfaz gráfica de escritorio (pywebview)
│   ├── app.py               #   arranque y endurecimiento de la ventana (CSP)
│   ├── bridge.py            #   API expuesta a JavaScript (sesión, rate limiting, validación)
│   └── web/                 #   index.html, styles.css, app.js
├── config.py                # Configuración global (rutas, límites, parámetros)
├── database.py              # Capa de datos: MongoDB (pymongo)
├── registro.py              # Guardado de perfiles (reglas de 4 a 25 muestras)
├── security.py              # PIN, bloqueo anti fuerza bruta, cifrado, validación, logs
├── audio_processor.py       # Captura de audio, extracción de embeddings, comparación
├── matching.py              # Lógica de decisión de identificación
├── setup_seguridad.py       # Genera PIN y clave de cifrado en .env
├── migrar_sqlite_a_mongo.py # Migración única desde la versión SQLite anterior
├── requirements.txt         # Dependencias (versiones fijadas y auditadas)
├── requirements-dev.txt     # + pytest, mongomock, pip-audit
└── data/                    # Se crea automáticamente (fuera de git)
    ├── audio/               #   audios .wav cifrados (nombres aleatorios)
    └── logs/                #   voiceid.log (errores técnicos, sin datos personales)
```

## 2. Preparar el entorno en VS Code

Abre una terminal en VS Code (`Ctrl + ñ` / `Ctrl + \``) dentro de la carpeta
`voice_identifier` y ejecuta. Requiere **Python 3.12** (numpy 1.26.4 no
tiene versión para 3.13):

### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_seguridad.py
```

### macOS / Linux
```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
python setup_seguridad.py
```

En VS Code, selecciona el intérprete del entorno virtual: `Ctrl+Shift+P` →
**"Python: Select Interpreter"** → elige el que apunta a `.venv312`.

### Dependencia del sistema operativo (importante)

`sounddevice` usa **PortAudio** por debajo. En la mayoría de sistemas ya viene
incluido, pero si al ejecutar el programa aparece un error relacionado con
PortAudio:

- **Windows**: normalmente no requiere nada adicional.
- **macOS**: `brew install portaudio`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install portaudio19-dev`

### MongoDB (base de datos)

La aplicación necesita un servidor MongoDB:

- **Local**: instala *MongoDB Community Server* (https://www.mongodb.com/try/download/community)
  como servicio de Windows. Escuchará en `127.0.0.1:27017`, que es el valor por defecto.
- **Docker**: `docker run -d --name voiceid-mongo -p 127.0.0.1:27017:27017 mongo:7`
- **Remoto (Atlas)**: pon la URI en `.env` como `VOICE_ID_MONGODB_URI=mongodb+srv://...`.
  Una conexión remota sin TLS se rechaza.

La configuración va en `.env` (ver `.env.example`); nunca en el código.

Si tenías datos en la versión SQLite anterior, migra una sola vez:

```bash
python migrar_sqlite_a_mongo.py --borrar-sqlite
```

## 3. Ejecutar el programa

```bash
python -m frontend.app   # interfaz gráfica
python main.py           # menú en terminal
```

Cada persona debe tener **entre 4 y 25 muestras de voz**: el registro pide 4
obligatorias y permite grabar más (opcionales) hasta 25. Desde *Gestionar
personas* se pueden re-entrenar (reemplazar todas) o añadir muestras sin pasar
del máximo. El límite se aplica en la interfaz, en `database.py` y en un
validador `$jsonSchema` de la propia colección de MongoDB.

## 4. Modelo de datos (MongoDB)

Colección `speakers`, un documento por persona con sus muestras embebidas
(toda escritura es atómica sin necesitar transacciones):

```json
{
  "nombre": "Ana María", "nombre_clave": "ana maría", "fecha_registro": "ISODate",
  "muestras": [
    {"archivo_audio": "<uuid>.wav.enc", "embedding": "BinData (cifrado)",
     "version_embedding": 2, "dim": 156, "fecha_creacion": "ISODate"}
  ]
}
```

Colección `estado_seguridad`: contador de intentos fallidos de PIN y bloqueo.

## 5. Seguridad

| # | Riesgo | Medida |
|---|--------|--------|
| 1 | Inyección SQL / NoSQL | Sin SQL. Los filtros de MongoDB solo reciben `str` validados; un objeto como `{"$ne": ""}` se rechaza antes de llegar a la BD. |
| 2 | Inyección de comandos | No se ejecutan comandos del sistema (`subprocess`, `os.system`, `eval`). |
| 3 | XSS | Todo dato dinámico pasa por `esc()` o `textContent`; CSP con nonce: un `<script>` inyectado no se ejecuta. |
| 4 | CSRF | No hay servidor HTTP ni cookies; la API solo existe dentro de la ventana. `form-action 'none'`. |
| 5 | SSRF | La app no hace peticiones a URLs; la URI de MongoDB solo viene del `.env`. `connect-src 'none'`. |
| 6 | Fallos de autenticación | PIN con PBKDF2-SHA256 (600 000 iteraciones), comparación en tiempo constante, PIN mínimo de 6 caracteres no trivial. |
| 7, 8 | Autorización / escalada | Denegar por defecto: todo método exige sesión salvo 3 de solo lectura; los métodos internos son privados (`_`); los nombres que empiezan por `_` están reservados. |
| 9, 21 | Exposición de información / trazas | Al usuario solo se muestran mensajes genéricos; los detalles van a `data/logs/voiceid.log`, sin nombres ni secretos. |
| 10 | Credenciales en el código | PIN, clave y URI solo en `.env` (fuera de git, permisos 600). |
| 11 | Sesiones | Se bloquea tras 10 min de inactividad o 8 h; botón «Bloquear sesión»; al bloquear se borran de memoria los audios pendientes. |
| 12 | Almacenamiento de contraseñas | Solo sal + hash PBKDF2 en `.env`; nunca el PIN. |
| 13 | Fuerza bruta | 3 fallos → bloqueo de 60 s, que se duplica en cada bloqueo seguido (máx. 1 h); se guarda en MongoDB, así que reiniciar no lo evita. |
| 14, 29 | Rate limiting / DoS | Límite de llamadas por método; grabación con tope de 30 s en memoria; tiempos de espera en MongoDB; máx. 25 muestras por persona. |
| 15, 16 | Archivos / Path Traversal | Nombres de archivo aleatorios generados por la app; se validan con un patrón y contra `data/audio` (también en el `$jsonSchema`). |
| 17 | Permisos / configuración | Carpetas `data/` 700, `.env` 600, sin herramientas de desarrollo, sin acceso a `file://`, modo privado. |
| 18 | Dependencias | Versiones fijadas; `pip-audit` sin vulnerabilidades conocidas. |
| 19, 20, 26 | CORS / HTTPS / cabeceras | La página se carga en memoria: no se abre el servidor HTTP de pywebview. MongoDB remoto exige TLS con certificados válidos. |
| 22 | Redirecciones | Si la ventana navega a una página externa, la sesión se bloquea y se recarga la app. |
| 23 | Deserialización | `np.load(..., allow_pickle=False)`; datos cifrados y autenticados (Fernet/HMAC); sin `pickle`. |
| 24, 25 | APIs / validación | Validación de tipo, formato y longitud de cada parámetro de la API JS. |
| 27 | Clickjacking | La app es una ventana de escritorio; `frame-src 'none'` y bloqueo si se carga dentro de un marco. |
| 28 | Copias y configuración | `.gitignore` excluye `.env*`, `*.db`, volcados, `*.bak`, logs; la migración ofrece borrar la base SQLite antigua. |
| 30 | Datos personales | Audio y embeddings cifrados (AES-128 + HMAC); los nombres de archivo no revelan identidades; borrado completo al eliminar a una persona. |

Ejecutar las pruebas: `pip install -r requirements-dev.txt` y luego `python -m pytest test`.
Auditar dependencias: `pip-audit -r requirements.txt`.

## 6. Notas técnicas

- **Embedding de voz**: MFCC + deltas, F0, formantes (LPC), energía,
  contraste espectral, ZCR y ritmo (`audio_processor.extraer_embedding`).
- **Umbral de decisión**: `UMBRAL_SIMILITUD = 0.995` en `config.py` (similitud
  coseno). Todos los embeddings tienen la misma norma, así que equivale a una
  distancia euclidiana ≤ 0,30, que se muestra solo como dato informativo. Con
  0,90 se aceptaba a personas distintas: si lo ajustas, calíbralo con voces reales.
