"""
main.py
-------
Punto de entrada del sistema de identificación de personas por voz (Fase 1).

Menú principal (exactamente 7 opciones):
    1) Registrar nueva persona (enrollment): graba varias muestras de voz.
    2) Identificar una voz: graba audio y lo compara contra todos los perfiles.
    3) Eliminar persona registrada (acción directa, en caliente).
    4) Gestionar personas (ver / renombrar / re-entrenar).
    5) Analizar modulación de voz en tiempo real (tono + volumen).
    6) Ejecutar pruebas automáticas de verificación (mic, BD, modelo).
    7) Salir.

Ejecución (desde la terminal integrada de VS Code, con el entorno
virtual activado):

    python main.py
"""

import sys
import random
import numpy as np

import config
import database as db
import audio_processor as ap
import security


# ---------------------------------------------------------------------------
# Duración de grabación adaptada a la longitud del texto a leer
# ---------------------------------------------------------------------------
def _duracion_para_texto(texto: str) -> int:
    """
    Calcula una duración de grabación generosa, adaptada a la longitud del
    texto (entre config.REGISTRO_DURACION_MIN y ..._MAX segundos), para
    que el usuario nunca se sienta apurado leyendo la frase completa.
    """
    num_palabras = len(texto.split())
    estimado = num_palabras / config.PALABRAS_POR_SEGUNDO_LECTURA + config.MARGEN_LECTURA_SEG
    return int(max(config.REGISTRO_DURACION_MIN, min(config.REGISTRO_DURACION_MAX, round(estimado))))


# ---------------------------------------------------------------------------
# Captura de UNA muestra con texto guiado + cuenta regresiva + barra en vivo
# + validación (calidad/SNR) + reproducción de verificación + reintento
# ---------------------------------------------------------------------------
def _capturar_muestra_valida(nombre: str, paso: int, total_pasos: int, texto_lectura: str):
    """
    Graba UNA muestra (un "paso" del registro/re-entrenamiento guiado):
    muestra el texto a leer, cuenta regresiva 3-2-1, graba con la barra de
    tono+volumen en vivo (reutilizando grabar_audio_con_analisis), valida
    su calidad (silencio/saturación/SNR) y ofrece escucharla antes de
    confirmarla.

    Manejo de errores QUIRÚRGICO: si la muestra se rechaza o el usuario no
    queda conforme al escucharla, se ofrece reintentar EN ESTE MISMO PASO —
    nunca se reinician los pasos ya completados ni se vuelve al menú.

    Devuelve (ruta_audio, embedding), o None si el usuario decide cancelar
    el reintento (las muestras de pasos anteriores, ya guardadas, se
    conservan tal cual).
    """
    duracion = _duracion_para_texto(texto_lectura)

    while True:
        print(f"\n--- Paso {paso}/{total_pasos} ---")
        print(f"📖 Lee en voz alta el siguiente texto (tendrás {duracion}s, sin prisa):\n   \"{texto_lectura}\"")
        input("\nPresiona ENTER cuando estés listo...")
        ap.cuenta_regresiva()
        audio, _, _, _ = ap.grabar_audio_con_analisis(duracion=duracion, guardar_audio=True)

        es_valida, motivo = ap.validar_calidad_muestra(audio)
        if not es_valida:
            print(f"⚠️  Muestra rechazada: {motivo}")
            reintentar = input("¿Repetir este paso ahora mismo? (s/n): ").strip().lower()
            if reintentar != "s":
                return None
            continue

        escuchar = input("¿Escuchar la grabación para verificarla? (s/n): ").strip().lower()
        if escuchar == "s":
            ap.reproducir_audio(audio)

        conservar = input("¿Conservar esta muestra? (s/n): ").strip().lower()
        if conservar != "s":
            continue  # vuelve a grabar el mismo paso, sin afectar los anteriores

        ruta_audio = ap.guardar_wav(audio, nombre_base=nombre)
        embedding = ap.extraer_embedding(audio)
        return ruta_audio, embedding


# ---------------------------------------------------------------------------
# OPCIÓN 1: Registro guiado (enrollment) de una nueva persona — 4 pasos fijos
# ---------------------------------------------------------------------------
def registrar_persona() -> None:
    nombre = input("Nombre de la persona a registrar: ").strip()
    if not nombre:
        print("❌ El nombre no puede estar vacío.")
        return

    session = db.SessionLocal()
    try:
        speaker = db.obtener_o_crear_speaker(session, nombre)

        print(f"\nRegistro guiado: {config.NUM_PASOS_REGISTRO} pasos, cada uno con un texto distinto para leer.")
        for paso in range(1, config.NUM_PASOS_REGISTRO + 1):
            texto = config.TEXTOS_LECTURA_REGISTRO[(paso - 1) % len(config.TEXTOS_LECTURA_REGISTRO)]
            resultado = _capturar_muestra_valida(nombre, paso, config.NUM_PASOS_REGISTRO, texto)
            if resultado is None:
                print(
                    f"\n⏹️  Registro detenido en el paso {paso}/{config.NUM_PASOS_REGISTRO}. "
                    f"Las muestras ya guardadas de '{nombre}' se conservan."
                )
                return

            ruta_audio, embedding = resultado
            db.guardar_muestra(session, speaker, ruta_audio, embedding)
            print(f"   ✅ Paso {paso}/{config.NUM_PASOS_REGISTRO} guardado en: {ruta_audio}")

        print(f"\n✅ Registro completo para '{nombre}' ({config.NUM_PASOS_REGISTRO} muestras).")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# OPCIÓN 2: Identificación de una voz nueva contra los perfiles guardados
# ---------------------------------------------------------------------------
def identificar_persona() -> None:
    session = db.SessionLocal()
    try:
        perfiles = db.obtener_embeddings_todos(session)
    finally:
        session.close()

    if not perfiles:
        print("⚠️  No hay personas registradas todavía. Usa la opción 1 primero.")
        return

    texto = random.choice(config.TEXTOS_LECTURA_VERIFICACION)
    duracion = _duracion_para_texto(texto)
    print(f"\n📖 Lee en voz alta el siguiente texto (distinto al usado en el registro; tendrás {duracion}s):")
    print(f"   \"{texto}\"")
    input("\nPresiona ENTER cuando estés listo...")
    ap.cuenta_regresiva()
    audio, _, _, _ = ap.grabar_audio_con_analisis(duracion=duracion, guardar_audio=True)

    es_valida, motivo = ap.validar_calidad_muestra(audio)
    if not es_valida:
        print(f"⚠️  No se pudo analizar la grabación: {motivo}")
        print("No se reconoce ninguna voz registrada")
        return

    embedding_nuevo = ap.extraer_embedding(audio)

    # --- Motor de comparación ESTRICTO: coseno Y euclidiana deben coincidir ---
    mejor_nombre, mejor_similitud, mejor_distancia = None, -1.0, float("inf")
    for nombre, embedding in perfiles:
        similitud = ap.similitud_coseno(embedding_nuevo, embedding)
        distancia = ap.distancia_euclidiana(embedding_nuevo, embedding)
        # Se elige el candidato con mayor similitud coseno (métrica principal);
        # la distancia euclidiana de ESE candidato se evalúa después como
        # segundo criterio obligatorio, no como desempate.
        if similitud > mejor_similitud:
            mejor_similitud, mejor_distancia, mejor_nombre = similitud, distancia, nombre

    print(f"\nMejor candidato: '{mejor_nombre}'  |  similitud coseno: {mejor_similitud:.4f}  |  distancia euclidiana: {mejor_distancia:.4f}")

    coincide_coseno = mejor_similitud >= config.UMBRAL_SIMILITUD
    coincide_euclidiana = mejor_distancia <= config.UMBRAL_DISTANCIA_EUCLIDIANA

    if coincide_coseno and coincide_euclidiana:
        print(f"🟢 IDENTIFICADO como: {mejor_nombre} (confianza {mejor_similitud:.1%})")

        vivacidad = ap.calcular_puntaje_vivacidad(audio)
        if vivacidad < config.VIVACIDAD_UMBRAL:
            print(
                f"⚠️  Aviso informativo: puntaje de vivacidad bajo ({vivacidad:.2f}). "
                "Esta es solo una heurística ligera, no una prueba de que el audio sea "
                "una grabación reproducida — no se bloquea el acceso, pero si tienes dudas, verifica en persona."
            )
    else:
        print("No se reconoce ninguna voz registrada")


# ---------------------------------------------------------------------------
# OPCIÓN 3: Listado de personas registradas
# ---------------------------------------------------------------------------
def listar_personas() -> None:
    session = db.SessionLocal()
    try:
        speakers = db.listar_speakers(session)
        if not speakers:
            print("⚠️  No hay personas registradas.")
            return

        print("\nPersonas registradas:")
        for s in speakers:
            print(
                f"  - {s.nombre}  (id={s.id}, muestras={len(s.muestras)}, "
                f"desde={s.fecha_registro:%Y-%m-%d %H:%M})"
            )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# OPCIÓN 4 (NUEVA): ANÁLISIS DE MODULACIÓN DE VOZ EN TIEMPO REAL (F0)
# ---------------------------------------------------------------------------
def _pedir_duracion_tono() -> int:
    """
    Pide la duración del análisis de tono, validando estrictamente que esté
    entre config.TONO_DURACION_MIN y config.TONO_DURACION_MAX (7-15s).

    Ligero a propósito: un solo bucle `while True` con un `try/except`
    puntual para capturar entradas no numéricas (letras, vacío, símbolos)
    sin que el programa se caiga. Reintenta EN EL MISMO LUGAR — nunca
    regresa al menú principal — hasta recibir un valor válido.
    """
    while True:
        entrada = input(
            f"Duración del análisis en segundos ({config.TONO_DURACION_MIN}-{config.TONO_DURACION_MAX}): "
        ).strip()

        try:
            duracion = int(entrada)
        except ValueError:
            print("⚠️  Ingresa solo un número entero (sin letras ni decimales).")
            continue

        if duracion > config.TONO_DURACION_MAX:
            print("El tiempo máximo para identificación de la voz es de máximo 15 segundos")
            continue

        if duracion < config.TONO_DURACION_MIN:
            print("El valor tiene que estar entre 7 y 15 segundos")
            continue

        return duracion


def analizar_modulacion_voz() -> None:
    """
    Graba audio del micrófono y, mientras la persona habla, muestra en la
    terminal dos barras en vivo: el tono (grave/agudo) y el nivel de
    volumen. Al finalizar, imprime un resumen con ambos promedios.
    """
    duracion = _pedir_duracion_tono()

    input(f"\nPresiona ENTER y luego habla durante {duracion} segundos para analizar tu voz...")
    audio, f0_promedio, clasificacion, volumen_promedio = ap.grabar_audio_con_analisis(duracion=duracion)

    print("\n" + "-" * 50)
    if f0_promedio > 0:
        print(f"Frecuencia fundamental promedio (F0): {f0_promedio:.1f} Hz")
        print(f"Clasificación de tu voz: {clasificacion}")
        print(f"Volumen promedio (RMS): {volumen_promedio:.4f}")
    else:
        print("⚠️  No se detectó suficiente señal de voz para analizar el tono.")
        print("   Revisa el micrófono o intenta hablar más fuerte y cerca.")
    print("-" * 50)


# ---------------------------------------------------------------------------
# OPCIÓN 6 (NUEVA): GESTIONAR PERSONAS REGISTRADAS (CRUD)
# ---------------------------------------------------------------------------
def _renombrar_persona_flujo() -> None:
    nombre_actual = input("\nNombre actual de la persona: ").strip()
    nombre_nuevo = input("Nuevo nombre: ").strip()
    if not nombre_actual or not nombre_nuevo:
        print("❌ Ambos nombres son obligatorios.")
        return

    session = db.SessionLocal()
    try:
        resultado = db.renombrar_speaker(session, nombre_actual, nombre_nuevo)
    finally:
        session.close()

    if resultado == "ok":
        print(f"✅ '{nombre_actual}' ahora se llama '{nombre_nuevo}'.")
    elif resultado == "no_existe":
        print(f"⚠️  No se encontró a '{nombre_actual}'.")
    elif resultado == "duplicado":
        print(f"❌ Ya existe una persona registrada como '{nombre_nuevo}'.")


def _reentrenar_persona_flujo() -> None:
    """
    Borra las muestras de voz actuales de una persona (audio + embedding,
    en BD y en disco) y graba muestras nuevas en su lugar, con el mismo
    registro guiado de 4 pasos (textos + validación) que el registro
    inicial. El Speaker (su identidad/nombre) no se toca, solo su
    "huella de voz".
    """
    nombre = input("\nNombre de la persona a re-entrenar: ").strip()
    if not nombre:
        print("❌ El nombre no puede estar vacío.")
        return

    session = db.SessionLocal()
    try:
        speaker = session.query(db.Speaker).filter_by(nombre=nombre).first()
        if speaker is None:
            print(f"⚠️  No se encontró a '{nombre}'.")
            return

        confirmar = input(
            f"Esto borrará las {len(speaker.muestras)} muestra(s) de voz actuales de "
            f"'{nombre}' y grabará {config.NUM_PASOS_REGISTRO} muestras nuevas (registro guiado). "
            f"¿Continuar? (s/n): "
        ).strip().lower()
        if confirmar != "s":
            print("Cancelado.")
            return

        borradas = db.eliminar_muestras_speaker(session, speaker)
        print(f"🗑️  {borradas} muestra(s) anterior(es) eliminada(s) de la BD y del disco.")

        for paso in range(1, config.NUM_PASOS_REGISTRO + 1):
            texto = config.TEXTOS_LECTURA_REGISTRO[(paso - 1) % len(config.TEXTOS_LECTURA_REGISTRO)]
            resultado = _capturar_muestra_valida(nombre, paso, config.NUM_PASOS_REGISTRO, texto)
            if resultado is None:
                print(f"\n⏹️  Re-entrenamiento detenido en el paso {paso}/{config.NUM_PASOS_REGISTRO}.")
                return

            ruta_audio, embedding = resultado
            db.guardar_muestra(session, speaker, ruta_audio, embedding)
            print(f"   ✅ Paso {paso}/{config.NUM_PASOS_REGISTRO} guardado en: {ruta_audio}")

        print(f"\n✅ Re-entrenamiento completo para '{nombre}'.")
    finally:
        session.close()


def _eliminar_persona_flujo() -> None:
    """
    Elimina a una persona por completo: de la base de datos Y de los
    archivos de audio/embedding en disco, con una transacción SQL directa
    y ligera (ver database.eliminar_speaker_completo). Pide confirmación
    escribiendo el nombre de nuevo, para evitar borrados accidentales.
    """
    nombre = input("\nNombre exacto de la persona a eliminar: ").strip()
    if not nombre:
        print("❌ El nombre no puede estar vacío.")
        return

    confirmacion = input(f"Escribe '{nombre}' de nuevo para confirmar el borrado: ").strip()
    if confirmacion != nombre:
        print("❌ Cancelado (el nombre no coincidió).")
        return

    existia, archivos_borrados = db.eliminar_speaker_completo(nombre)

    if existia:
        print(f"✅ '{nombre}' eliminado de la base de datos ({archivos_borrados} archivo(s) borrado(s) de disco).")
    else:
        print(f"⚠️  No se encontró a '{nombre}' en el sistema.")


def gestionar_personas() -> None:
    """Submenú para editar personas ya registradas (renombrar / re-entrenar)."""
    while True:
        print("\n" + "-" * 50)
        print("  GESTIONAR PERSONAS REGISTRADAS")
        print("-" * 50)
        print("1) Ver personas registradas")
        print("2) Renombrar una persona")
        print("3) Re-entrenar (reemplazar sus muestras de voz)")
        print("4) Volver al menú principal")
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "1":
            listar_personas()
        elif opcion == "2":
            _renombrar_persona_flujo()
        elif opcion == "3":
            _reentrenar_persona_flujo()
        elif opcion == "4":
            return
        else:
            print("❌ Opción no válida, intenta de nuevo.")


# ---------------------------------------------------------------------------
# OPCIÓN 7: PRUEBAS DE VERIFICACIÓN AUTOMÁTICAS (requerimiento 5)
# ---------------------------------------------------------------------------
def ejecutar_pruebas_automaticas() -> None:
    """
    Valida rápidamente los 3 componentes críticos del sistema:
        1) Micrófono: ¿se puede grabar audio y contiene señal real?
        2) Base de datos: ¿se puede escribir y releer un registro de prueba?
        3) Modelo/embedding: ¿se puede procesar la muestra y comparar
           correctamente (similitud consigo misma ≈ 1.0)?

    El registro '_usuario_prueba_' creado aquí es solo para diagnóstico y
    puede borrarse luego sin afectar a las personas reales registradas.
    """
    print("\n" + "=" * 60)
    print("🔍 EJECUTANDO PRUEBAS DE VERIFICACIÓN AUTOMÁTICAS")
    print("=" * 60)

    errores = []
    audio_test = None

    # --- Test 1: Micrófono ---
    print("\n[1/3] Probando captura de micrófono (2 segundos)...")
    try:
        audio_test = ap.grabar_audio(duracion=2)
        volumen = float(np.abs(audio_test).mean())
        if volumen < 1e-6:
            print("   ⚠️  Se grabó audio, pero parece silencio total (revisa el micrófono).")
        else:
            print(f"   ✅ Micrófono OK. Nivel de señal promedio: {volumen:.6f}")
    except Exception as e:
        errores.append(f"Micrófono: {e}")
        print(f"   ❌ Error al grabar: {e}")
        print("   ↳ Se usará una señal sintética para poder continuar con las demás pruebas.")
        audio_test = (np.random.randn(config.SAMPLE_RATE * 2) * 0.01).astype(np.float32)

    # --- Test 2: Base de datos ---
    print("\n[2/3] Probando conexión, escritura y lectura en la base de datos...")
    try:
        db.init_db()
        session = db.SessionLocal()
        try:
            speaker_prueba = db.obtener_o_crear_speaker(session, "_usuario_prueba_")
            ruta_wav = ap.guardar_wav(audio_test, nombre_base="_prueba_")
            embedding_prueba = ap.extraer_embedding(audio_test)
            db.guardar_muestra(session, speaker_prueba, ruta_wav, embedding_prueba)

            # Se relee desde la BD para confirmar que la persistencia fue real
            perfiles = db.obtener_embeddings_todos(session)
        finally:
            session.close()

        if any(nombre == "_usuario_prueba_" for nombre, _ in perfiles):
            print(f"   ✅ Base de datos OK. Registro de prueba guardado en: {config.DB_PATH}")
        else:
            raise RuntimeError("El registro de prueba no se encontró después de guardarlo.")
    except Exception as e:
        errores.append(f"Base de datos: {e}")
        print(f"   ❌ Error de base de datos: {e}")

    # --- Test 3: Extracción de embedding y comparación (modelo) ---
    print("\n[3/3] Probando extracción de embedding y comparación...")
    try:
        emb1 = ap.extraer_embedding(audio_test)
        emb2 = ap.extraer_embedding(audio_test)  # misma señal -> similitud esperada ≈ 1.0
        score = ap.similitud_coseno(emb1, emb2)
        if score > 0.99:
            print(f"   ✅ Modelo OK. Similitud de la muestra consigo misma: {score:.4f} (esperado ≈ 1.0000)")
        else:
            raise RuntimeError(f"Similitud inesperadamente baja: {score:.4f}")
    except Exception as e:
        errores.append(f"Modelo/embedding: {e}")
        print(f"   ❌ Error al procesar el embedding: {e}")

    # --- Resumen final ---
    print("\n" + "=" * 60)
    if not errores:
        print("🎉 TODAS LAS PRUEBAS PASARON CORRECTAMENTE. El sistema está listo para usarse.")
    else:
        print(f"⚠️  Se encontraron {len(errores)} problema(s):")
        for err in errores:
            print(f"   - {err}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# MENÚ PRINCIPAL
# ---------------------------------------------------------------------------
def menu() -> None:
    while True:
        print("\n" + "-" * 50)
        print("  SISTEMA DE IDENTIFICACIÓN DE VOZ - FASE 1")
        print("-" * 50)
        print("1) Registrar nueva persona (enrollment)")
        print("2) Identificar una voz")
        print("3) Eliminar persona registrada (acción directa)")
        print("4) Gestionar personas (ver / renombrar / re-entrenar)")
        print("5) Analizar modulación de voz en tiempo real (tono + volumen)")
        print("6) Ejecutar pruebas automáticas de verificación")
        print("7) Salir")
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "1":
            registrar_persona()
        elif opcion == "2":
            identificar_persona()
        elif opcion == "3":
            _eliminar_persona_flujo()
        elif opcion == "4":
            gestionar_personas()
        elif opcion == "5":
            analizar_modulacion_voz()
        elif opcion == "6":
            ejecutar_pruebas_automaticas()
        elif opcion == "7":
            print("Hasta luego 👋")
            sys.exit(0)
        else:
            print("❌ Opción no válida, intenta de nuevo.")


if __name__ == "__main__":
    db.init_db()
    if not security.pedir_acceso():
        sys.exit(1)
    print("⚙️  Preparando el motor de voz (una sola vez, esto puede tardar unos segundos)...")
    ap.precalentar_motor()
    menu()