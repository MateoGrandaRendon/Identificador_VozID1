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
├── calibrar_umbral.py       # Mide la identificación con las personas registradas
├── recalcular_embeddings.py # Recalcula los embeddings al cambiar de motor de voz
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
python -m motor_ecapa        # descarga única del modelo de voz (~80 MB)
python setup_seguridad.py
```

### macOS / Linux
```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
python -m motor_ecapa        # descarga única del modelo de voz (~80 MB)
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

> **Si el equipo tiene más de una cuenta de Windows (o lo usan otras personas), activa la
> autenticación de MongoDB.** Sin ella, cualquier cuenta o programa del equipo puede
> conectarse a `127.0.0.1:27017` y ver los nombres de las personas registradas, o borrar
> o renombrar sus perfiles (los embeddings siguen cifrados). Pasos:
> 1. Con `mongosh`, crea un usuario solo para la app:
>    `use voice_id` y luego
>    `db.createUser({user: "voiceid", pwd: passwordPrompt(), roles: ["readWrite", "dbAdmin"]})`.
> 2. En `mongod.cfg` (p. ej. `C:\Program Files\MongoDB\Server\<versión>\bin\mongod.cfg`) añade
>    `security:` / `  authorization: enabled` y reinicia el servicio *MongoDB*.
>    En Docker: crea el contenedor con `-e MONGO_INITDB_ROOT_USERNAME=... -e MONGO_INITDB_ROOT_PASSWORD=...`.
> 3. En `.env`: `VOICE_ID_MONGODB_URI=mongodb://voiceid:CLAVE@127.0.0.1:27017/voice_id`.

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
     "version_embedding": 3, "dim": 192, "fecha_creacion": "ISODate"}
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
| 7, 8 | Autorización / escalada | Denegar por defecto: todo método exige sesión salvo 3 de solo lectura. A pywebview se le entrega una fachada vacía y solo los métodos públicos se registran por nombre exacto (`window.expose`), porque su despachador alcanza cualquier atributo, incluso privados (`_`) y `__setattr__`. Los nombres que empiezan por `_` están reservados (también en la migración). |
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

- **Embedding de voz** (`config.MOTOR_EMBEDDING`):
  - `ecapa` (por defecto): modelo preentrenado ECAPA-TDNN de SpeechBrain
    (`motor_ecapa.py`), 192 dimensiones. Descarga única (~80 MB):
    `python -m motor_ecapa`. Revisión fijada, SHA-256 verificado y carga con
    `weights_only=True` (el archivo no puede ejecutar código).
  - `clasico`: MFCC + deltas, F0, formantes (LPC), energía, contraste, ZCR y
    ritmo. Con voces reales da 96-99 % entre cualquier par de voces, así que no
    separa a las personas; queda solo para equipos sin PyTorch
    (`VOICE_ID_MOTOR=clasico` en `.env`).
  - Al cambiar de motor, `python recalcular_embeddings.py` recalcula todos los
    embeddings desde los audios de `data/audio`, sin volver a grabar.
- **Decisión de identificación** (`matching.py`): cada persona se puntúa con el
  **promedio de sus 4 muestras más parecidas** (`MUESTRAS_COMPARADAS`), no con
  una sola muestra suelta. Se acepta si ese promedio alcanza el umbral
  (ECAPA: `0.40`, medido con voces reales: misma persona 63-77 %, otras
  personas ≤ 9 %) **y** la segunda persona está al menos
  `MARGEN_DISTANCIA = 1.1` veces más lejos en distancia euclidiana. Así, una voz
  que se parece por igual a dos personas no se asigna a ninguna.
- **Calibración**: `python calibrar_umbral.py` (pide el PIN) evalúa la regla con
  las personas ya registradas y recomienda umbral y margen. Hacen falta al menos
  2 personas; el resultado es optimista porque las muestras son de la misma sesión.
