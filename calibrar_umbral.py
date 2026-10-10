"""
calibrar_umbral.py
------------------
Mide qué tan bien funciona la regla de identificación (matching.py) con las
personas YA registradas y recomienda UMBRAL_SIMILITUD y MARGEN_DISTANCIA.
No graba nada ni modifica la base de datos ni config.py: solo informa.

    python calibrar_umbral.py

Cómo evalúa (validación "deja uno fuera", sin grabar audio nuevo):
  - Persona registrada: cada muestra de cada persona se compara contra todos
    los perfiles SIN esa muestra. Debe identificarse como su dueño.
  - Persona desconocida: cada muestra se compara contra los perfiles de las
    DEMÁS personas (como si su dueño no estuviera registrado). No debe
    identificarse como nadie.

Aviso: las muestras de registro se graban en una misma sesión, así que el
resultado es optimista. Para una cifra realista, registra a varias personas
y luego graba identificaciones en otro momento o lugar.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import config
import database as db
import matching
import security

MARGENES = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3)


@dataclass
class Resultado:
    umbral: float
    margen: float
    aciertos: float       # registradas identificadas como su dueño
    confusiones: float    # registradas identificadas como OTRA persona
    falsas: float         # desconocidas identificadas como alguien


def preparar_pruebas(perfiles_compatibles):
    """
    Devuelve (registradas, desconocidas): listas de (dueño, puntuaciones) ya
    calculadas con matching.puntuar_personas, para evaluar muchos umbrales
    sin volver a comparar vectores.
    """
    por_persona = defaultdict(list)
    for nombre, emb in perfiles_compatibles:
        por_persona[nombre].append(emb)
    registradas, desconocidas = [], []
    for nombre, muestras in por_persona.items():
        otras = [(n, e) for n, es in por_persona.items() if n != nombre for e in es]
        for i, muestra in enumerate(muestras):
            resto = [(nombre, e) for j, e in enumerate(muestras) if j != i]
            if resto:
                registradas.append((nombre, matching.puntuar_personas(muestra, otras + resto)))
            if otras:
                desconocidas.append((nombre, matching.puntuar_personas(muestra, otras)))
    return registradas, desconocidas


def evaluar(registradas, desconocidas, umbral: float, margen: float) -> Resultado:
    aciertos = confusiones = falsas = 0
    for dueño, puntuaciones in registradas:
        r = matching.decidir(puntuaciones, umbral, margen)
        aciertos += r.identificado and r.nombre == dueño
        confusiones += r.identificado and r.nombre != dueño
    for _, puntuaciones in desconocidas:
        falsas += matching.decidir(puntuaciones, umbral, margen).identificado
    n_reg, n_desc = max(len(registradas), 1), max(len(desconocidas), 1)
    return Resultado(umbral, margen, aciertos / n_reg, confusiones / n_reg, falsas / n_desc)


def umbrales_candidatos(registradas, n: int = 12) -> list[float]:
    """Umbrales repartidos entre las puntuaciones reales del dueño, más el configurado."""
    propias = [p.similitud for dueño, ps in registradas for p in ps if p.nombre == dueño]
    if not propias:
        return [config.UMBRAL_SIMILITUD]
    candidatos = np.quantile(propias, np.linspace(0.0, 0.9, n))
    return sorted({round(float(u), 4) for u in candidatos} | {config.UMBRAL_SIMILITUD})


def recomendar(resultados: list[Resultado]) -> Resultado:
    """
    Prioriza no identificar a nadie por error (falsas + confusiones = 0) y,
    entre esas opciones, la que más aciertos tiene; a igualdad, la más
    estricta. Si ninguna llega a cero errores, la de menor error total.
    """
    sin_errores = [r for r in resultados if r.falsas == 0 and r.confusiones == 0]
    if sin_errores:
        return max(sin_errores, key=lambda r: (r.aciertos, r.umbral, r.margen))
    return min(resultados, key=lambda r: (r.falsas + r.confusiones + (1 - r.aciertos), -r.umbral))


def _fila(r: Resultado, marca: str = "") -> str:
    return (f"  {r.umbral:>8.4f}  x{r.margen:<5.2f} | {100 * r.aciertos:>6.1f}%  {100 * r.confusiones:>6.1f}%"
            f"  {100 * r.falsas:>6.1f}%  {marca}")


def main() -> None:
    security.configurar_logs()
    try:
        security.verificar_clave_cifrado()
        db.init_db()
    except (db.BaseDatosError, RuntimeError) as e:
        print(e)
        sys.exit(1)
    if not security.pedir_acceso():     # descifra datos biométricos: exige el PIN
        sys.exit(1)

    compatibles, desactualizados = matching.filtrar_perfiles_compatibles(db.obtener_embeddings_todos())
    personas = {n for n, _ in compatibles}
    if len(personas) < 2:
        print("Hacen falta al menos 2 personas registradas (con muestras vigentes) para calibrar.")
        sys.exit(1)
    if desactualizados:
        print(f"Se omiten (re-entrénalas para incluirlas): {', '.join(desactualizados)}")

    print(f"Evaluando {len(compatibles)} muestras de {len(personas)} personas "
          f"(promedio de las {config.MUESTRAS_COMPARADAS} más parecidas)...")
    registradas, desconocidas = preparar_pruebas(compatibles)
    resultados = [evaluar(registradas, desconocidas, u, m)
                  for u in umbrales_candidatos(registradas) for m in MARGENES]
    actual = evaluar(registradas, desconocidas, config.UMBRAL_SIMILITUD, config.MARGEN_DISTANCIA)
    mejor = recomendar(resultados)

    print("\n    umbral  margen | aciertos  confus.  falsas")
    for r in resultados:
        print(_fila(r, "<- recomendado" if r == mejor else ""))
    print("\nConfiguración actual:")
    print(_fila(actual))
    print("\nRecomendación (config.py):")
    print(f"  UMBRAL_SIMILITUD = {mejor.umbral}")
    print(f"  MARGEN_DISTANCIA = {mejor.margen}")
    print("\naciertos = registradas reconocidas · confus. = confundidas con otra persona · "
          "falsas = desconocidas identificadas como alguien.")
    print("Resultado optimista: las muestras de registro son de la misma sesión (ver docstring).")


if __name__ == "__main__":
    main()
