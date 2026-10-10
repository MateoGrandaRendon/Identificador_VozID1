"""
test/test_calibracion.py
------------------------
Pruebas de calibrar_umbral.py con perfiles sintéticos (sin BD ni audio).
"""
import numpy as np

import calibrar_umbral as cal
import config


def _perfiles(centros: dict, muestras: int = 6, ruido: float = 0.02, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    return [(nombre, centro + ruido * rng.standard_normal(config.EMBEDDING_DIM))
            for nombre, centro in centros.items() for _ in range(muestras)]


def _centro(semilla):
    v = np.random.default_rng(semilla).standard_normal(config.EMBEDDING_DIM)
    return v / np.linalg.norm(v)


def test_personas_bien_separadas_se_reconocen_sin_errores():
    perfiles = _perfiles({"Ana": _centro(1), "Luis": _centro(2), "Marta": _centro(3)})
    registradas, desconocidas = cal.preparar_pruebas(perfiles)
    assert len(registradas) == 18 and len(desconocidas) == 18     # deja-uno-fuera y "no registrada"
    r = cal.evaluar(registradas, desconocidas, umbral=0.5, margen=1.1)
    assert (r.aciertos, r.confusiones) == (1.0, 0.0)


def test_el_margen_evita_confundir_a_dos_personas_casi_iguales():
    """Luis y su "gemelo" tienen casi la misma voz: sin margen, una muestra de
    uno se asigna a menudo al otro; con margen el caso ambiguo no se asigna a
    nadie. (Un desconocido idéntico a alguien registrado no se puede
    distinguir con ningún margen: eso no lo arregla esta regla.)"""
    luis = _centro(2)
    perfiles = _perfiles({"Ana": _centro(1), "Luis": luis, "Gemelo": luis + 0.001 * _centro(9)})
    registradas, desconocidas = cal.preparar_pruebas(perfiles)
    sin_margen = cal.evaluar(registradas, desconocidas, umbral=0.5, margen=1.0)
    con_margen = cal.evaluar(registradas, desconocidas, umbral=0.5, margen=1.2)
    assert sin_margen.confusiones > 0
    assert con_margen.confusiones == 0


def test_recomendar_prefiere_cero_errores_y_luego_mas_aciertos():
    resultados = [
        cal.Resultado(0.99, 1.0, aciertos=1.0, confusiones=0.0, falsas=0.3),
        cal.Resultado(0.99, 1.1, aciertos=0.8, confusiones=0.0, falsas=0.0),
        cal.Resultado(0.995, 1.1, aciertos=0.6, confusiones=0.0, falsas=0.0),
    ]
    assert cal.recomendar(resultados) == resultados[1]


def test_recomendar_sin_opcion_perfecta_minimiza_el_error():
    resultados = [
        cal.Resultado(0.99, 1.0, aciertos=1.0, confusiones=0.2, falsas=0.5),
        cal.Resultado(0.99, 1.2, aciertos=0.7, confusiones=0.0, falsas=0.1),
    ]
    assert cal.recomendar(resultados) == resultados[1]


def test_umbrales_candidatos_incluyen_el_configurado():
    perfiles = _perfiles({"Ana": _centro(1), "Luis": _centro(2)})
    registradas, _ = cal.preparar_pruebas(perfiles)
    assert config.UMBRAL_SIMILITUD in cal.umbrales_candidatos(registradas)
