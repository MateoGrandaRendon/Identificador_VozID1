# Sistema de Identificación de Voz — Fase 1

Identificación de personas por su voz (Speaker Identification), no
reconocimiento de lo que dicen (eso sería speech-to-text).

## 1. Estructura del proyecto

```
voice_identifier/
├── main.py              # Punto de entrada: menú CLI, enrollment, identificación, pruebas
├── config.py            # Configuración global (rutas y parámetros)
├── database.py          # Capa de datos: SQLite + SQLAlchemy (Speaker, VoiceSample)
├── audio_processor.py   # Captura de audio, extracción de embeddings, comparación
├── requirements.txt     # Dependencias
├── README.md
└── data/                # Se crea automáticamente al ejecutar
    ├── voice_id.db       # Base de datos SQLite
    └── audio/            # Muestras .wav y embeddings .npy
```

## 2. Preparar el entorno en VS Code

Abre una terminal en VS Code (`Ctrl + ñ` / `Ctrl + \``) dentro de la carpeta
`voice_identifier` y ejecuta:

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En VS Code, selecciona el intérprete del entorno virtual: `Ctrl+Shift+P` →
**"Python: Select Interpreter"** → elige el que apunta a `.venv`.

### Dependencia del sistema operativo (importante)

`sounddevice` usa **PortAudio** por debajo. En la mayoría de sistemas ya viene
incluido, pero si al ejecutar el programa aparece un error relacionado con
PortAudio:

- **Windows**: normalmente no requiere nada adicional.
- **macOS**: `brew install portaudio`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install portaudio19-dev`

## 3. Ejecutar el programa

Con el entorno virtual activado:

```bash
python main.py
```

Aparecerá un menú:

```
1) Registrar nueva persona (enrollment)
2) Identificar una voz
3) Listar personas registradas
4) Ejecutar pruebas automáticas de verificación
5) Salir
```

### Flujo recomendado la primera vez

1. Opción **4** — corre las pruebas automáticas para confirmar que el
   micrófono, la base de datos y el modelo funcionan en tu máquina.
2. Opción **1** — registra a una persona (te pedirá hablar 4 veces, 4
   segundos cada vez).
3. Opción **2** — habla de nuevo y verifica que el sistema te identifica
   correctamente.
4. Opción **3** — revisa el listado de personas y cuántas muestras tiene cada una.

## 4. Notas técnicas (Fase 1)

- **Embedding de voz**: se calcula con MFCC (`librosa`) + media/desviación
  estándar, normalizado con norma L2. Es rápido y no requiere descargar
  modelos pesados, ideal para validar el flujo completo. En una fase
  posterior se puede sustituir por un modelo de deep learning
  (por ejemplo, `resemblyzer` o SpeechBrain ECAPA-TDNN) cambiando solo la
  función `extraer_embedding` en `audio_processor.py` — el resto del
  sistema (base de datos, comparación, CLI) no necesita cambiar.
- **Umbral de decisión**: `UMBRAL_SIMILITUD` en `config.py` (por defecto
  0.80). Si el sistema identifica mal, prueba subiendo o bajando ese valor.
- **Base de datos**: SQLite vía SQLAlchemy. El archivo vive en
  `data/voice_id.db` y se puede abrir con cualquier visor de SQLite
  (por ejemplo, la extensión "SQLite Viewer" de VS Code) para inspeccionar
  las tablas `speakers` y `voice_samples`.
- **Preparado para crecer**: la separación en módulos (`config`,
  `database`, `audio_processor`, `main`) permite conectar después una
  interfaz web (HTML/CSS/JS) que llame a estas mismas funciones desde una
  API (por ejemplo con FastAPI o Flask), sin reescribir la lógica de
  audio ni de base de datos.
