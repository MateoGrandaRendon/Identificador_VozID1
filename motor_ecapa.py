"""
motor_ecapa.py
--------------
Embedding de hablante con ECAPA-TDNN, el modelo preentrenado de SpeechBrain
"spkrec-ecapa-voxceleb" (entrenado con miles de voces de VoxCeleb). Sustituye
al embedding de características hechas a mano, que daba similitudes de
0,96-0,99 entre CUALQUIER par de voces y no permitía separar personas.

Seguridad del modelo (un checkpoint de PyTorch puede ejecutar código al
cargarse, y el YAML de SpeechBrain puede instanciar objetos arbitrarios):
  - Se descarga una revisión FIJA del repositorio (commit) y se comprueba el
    SHA-256 del archivo antes de cada carga.
  - Se carga con torch.load(weights_only=True): solo tensores, nunca código.
  - No se usa hyperparams.yaml: la arquitectura se construye aquí, con los
    mismos parámetros que ese archivo.

Descarga única del modelo (~80 MB):  python -m motor_ecapa
"""

import hashlib
import threading
from pathlib import Path

import numpy as np

import config

REPO = "speechbrain/spkrec-ecapa-voxceleb"
REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ARCHIVO = "embedding_model.ckpt"
SHA256 = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
DIM = 192
SAMPLE_RATE = 16000   # el modelo se entrenó a 16 kHz


class MotorNoDisponible(RuntimeError):
    """El modelo no está descargado o no coincide con la versión verificada (mensaje apto para el usuario)."""


_modelo = None
_fbank = None
_lock = threading.Lock()


def ruta_modelo() -> Path:
    return config.MODELO_DIR / ARCHIVO


def _verificar(ruta: Path) -> None:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    if h.hexdigest() != SHA256:
        raise MotorNoDisponible(
            f"El modelo de voz ({ruta}) no coincide con la versión verificada (archivo dañado o alterado). "
            "Bórralo y vuelve a descargarlo con: python -m motor_ecapa"
        )


def descargar_modelo() -> Path:
    """Descarga la revisión fijada del modelo y verifica su SHA-256."""
    from huggingface_hub import hf_hub_download
    ruta = Path(hf_hub_download(REPO, ARCHIVO, revision=REVISION, local_dir=config.MODELO_DIR))
    _verificar(ruta)
    return ruta


def _cargar():
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    ruta = ruta_modelo()
    if not ruta.is_file():
        raise MotorNoDisponible("Falta el modelo de voz ECAPA. Descárgalo una vez con: python -m motor_ecapa")
    _verificar(ruta)
    modelo = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                        dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=DIM)
    modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
    return modelo.eval(), Fbank(n_mels=80)


def verificar() -> None:
    """Carga el modelo (si aún no lo estaba). Lanza MotorNoDisponible con un mensaje claro si falta o está alterado."""
    global _modelo, _fbank
    with _lock:
        if _modelo is None:
            _modelo, _fbank = _cargar()


def extraer_embedding(audio: np.ndarray, samplerate: int = None) -> np.ndarray:
    """Vector de 192 dimensiones normalizado (norma 1) que caracteriza la voz."""
    import torch

    verificar()
    señal = np.asarray(audio, dtype=np.float32).reshape(-1)
    samplerate = samplerate or config.SAMPLE_RATE
    if samplerate != SAMPLE_RATE:
        import librosa
        señal = librosa.resample(señal, orig_sr=samplerate, target_sr=SAMPLE_RATE)
    with _lock, torch.no_grad():
        feats = _fbank(torch.from_numpy(señal).unsqueeze(0))
        feats = feats - feats.mean(dim=1, keepdim=True)   # normalización por frase (norm_type: sentence)
        emb = _modelo(feats).reshape(-1).numpy().astype(np.float64)
    norma = np.linalg.norm(emb)
    return emb / norma if norma > 0 else emb


if __name__ == "__main__":
    print(f"Descargando {REPO} (revisión {REVISION[:12]})...")
    print(f"Modelo verificado en: {descargar_modelo()}")
