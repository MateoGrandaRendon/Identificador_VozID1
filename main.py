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

import logging
import sys
import random
import numpy as np

import config
import database as db
import audio_processor as ap
import matching
import registro
import security

log = logging.getLogger("voiceid.cli")


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
def _capturar_muestra_valida(paso: int, etiqueta: str, texto_lectura: str):
    """
    Graba UNA muestra (un "paso" del registro/re-entrenamiento guiado):
    muestra el texto a leer, cuenta regresiva 3-2-1, graba con la barra de
    tono+volumen en vivo (reutilizando grabar_audio_con_analisis), valida
    su calidad (silencio/saturación/SNR) y ofrece escucharla antes de
    confirmarla.

    Manejo de errores QUIRÚRGICO: si la muestra se rechaza o el usuario no
    queda conforme al escucharla, se ofrece reintentar EN ESTE MISMO PASO —
    nunca se reinician los pasos ya completados ni se vuelve al menú.

    Devuelve el audio (np.ndarray) o None si el usuario cancela. No guarda
    nada: el perfil se persiste completo al final (registro.guardar_perfil).
    """
    duracion = _duracion_para_texto(texto_lectura)

    while True:
        print(f"\n--- Muestra {paso} ({etiqueta}) ---")
        print(f" Lee en voz alta el siguiente texto (tendrás {duracion}s, sin prisa):\n   \"{texto_lectura}\"")
        input("\nPresiona ENTER cuando estés listo...")
        ap.cuenta_regresiva()
        audio, _, _, _ = ap.grabar_audio_con_analisis(duracion=duracion, guardar_audio=True)

        es_valida, motivo = ap.validar_calidad_muestra(audio)
        if not es_valida:
            print(f"  Muestra rechazada: {motivo}")
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

        return audio


def _pedir_nombre(mensaje: str):
    """Pide y valida un nombre de persona. Devuelve None si no es válido."""
    try:
        return security.validar_nombre_persona(input(mensaje))
    except ValueError as e:
        print(f" {e}")
        return None


def _grabar_muestras(minimo: int, maximo: int):
    """
    Graba entre `minimo` (obligatorias) y `maximo` muestras: tras las
    obligatorias, pregunta si se quieren añadir más (hasta el máximo).
    Devuelve la lista de audios, o None si se cancela antes del mínimo.
    """
    audios = []
    while len(audios) < maximo:
        paso = len(audios) + 1
        if paso > minimo:
            otra = input(
                f"\n¿Grabar otra muestra opcional? ({len(audios)} de máx. {maximo}) (s/n): "
            ).strip().lower()
            if otra != "s":
                break
        texto = config.TEXTOS_LECTURA_REGISTRO[(paso - 1) % len(config.TEXTOS_LECTURA_REGISTRO)]
        etiqueta = f"obligatoria {paso}/{minimo}" if paso <= minimo else f"opcional, máx. {maximo}"
        audio = _capturar_muestra_valida(paso, etiqueta, texto)
        if audio is None:
            if len(audios) < minimo:
                return None
            break
        audios.append(audio)
    return audios


def _guardar_perfil_cli(nombre: str, audios, modo: str) -> None:
    print("\n  Procesando y guardando (extrayendo embeddings)...")
    try:
        n = registro.guardar_perfil(nombre, audios, modo)
    except db.BaseDatosError as e:
        print(f" {e}")
        return
    print(f" {n} muestra(s) guardada(s) para '{nombre}'.")


# ---------------------------------------------------------------------------
# OPCIÓN 1: Registro guiado (enrollment) de una nueva persona — 4 a 25 muestras
# ---------------------------------------------------------------------------
def registrar_persona() -> None:
    nombre = _pedir_nombre("Nombre de la persona a registrar: ")
    if nombre is None:
        return
    if db.obtener_speaker(nombre) is not None:
        print(f" '{nombre}' ya existe. Usa 'Gestionar personas' para re-entrenar o añadir muestras.")
        return

    minimo, maximo = registro.limites_nuevas_muestras("registro")
    print(
        f"\nRegistro guiado: {minimo} muestras obligatorias (cada una con un texto distinto) "
        f"y, si quieres, más muestras opcionales hasta {maximo}. No se guarda nada hasta el final."
    )
    audios = _grabar_muestras(minimo, maximo)
    if audios is None:
        print(f"\n  Registro cancelado: no se guardó nada (se necesitan al menos {minimo} muestras).")
        return
    _guardar_perfil_cli(nombre, audios, "registro")


# ---------------------------------------------------------------------------
# OPCIÓN 2: Identificación de una voz nueva contra los perfiles guardados
# ---------------------------------------------------------------------------
def identificar_persona() -> None:
    perfiles_todos = db.obtener_embeddings_todos()

    if not perfiles_todos:
        if db.contar_speakers():
            print("  Hay personas registradas, pero sus datos no se pudieron descifrar: "
                  "revisa que VOICE_ID_FERNET_KEY en .env sea la clave original.")
        else:
            print("  No hay personas registradas todavía. Usa la opción 1 primero.")
        return

    # Solo se comparan embeddings generados con la versión Y dimensión
    # VIGENTES del pipeline (matching.filtrar_perfiles_compatibles). Esto
    # es lo que evita el error "shapes (156,) and (60,) not aligned":
    # nunca se intenta comparar un embedding viejo contra uno nuevo.
    perfiles, desactualizados = matching.filtrar_perfiles_compatibles(perfiles_todos)

    if desactualizados:
        print(
            f"  Aviso: {len(desactualizados)} persona(s) tienen muestras de una versión "
            f"anterior del motor y no se incluyen en esta comparación: {', '.join(desactualizados)}. "
            "Usa 'Gestionar personas → Re-entrenar' para actualizarlas."
        )

    if not perfiles:
        print("  No hay perfiles compatibles con la versión actual del motor. Registra o re-entrena a alguien primero.")
        return

    texto = random.choice(config.TEXTOS_LECTURA_VERIFICACION)
    duracion = _duracion_para_texto(texto)
    print(f"\n Lee en voz alta el siguiente texto (distinto al usado en el registro; tendrás {duracion}s):")
    print(f"   \"{texto}\"")
    input("\nPresiona ENTER cuando estés listo...")
    ap.cuenta_regresiva()
    audio, _, _, _ = ap.grabar_audio_con_analisis(duracion=duracion, guardar_audio=True)

    es_valida, motivo = ap.validar_calidad_muestra(audio)
    if not es_valida:
        print(f"  No se pudo analizar la grabación: {motivo}")
        print("No se reconoce ninguna voz registrada")
        return

    embedding_nuevo = ap.extraer_embedding(audio)

    # --- Promedio de las 4 muestras más parecidas + umbral + margen frente al 2º (matching.py) ---
    # (lógica de decisión centralizada en matching.py, compartida con la GUI)
    resultado = matching.identificar_mejor_candidato(embedding_nuevo, perfiles)

    print(
        f"\nMejor candidato: '{resultado.nombre}'  |  similitud coseno (media top-{config.MUESTRAS_COMPARADAS}): "
        f"{resultado.similitud:.4f}  |  distancia euclidiana media: {resultado.distancia:.4f}"
    )
    if resultado.margen is not None:
        print(
            f"Segundo candidato: '{resultado.segundo}'  |  margen: x{resultado.margen:.2f} "
            f"(mínimo x{config.MARGEN_DISTANCIA:.2f})"
        )

    if resultado.identificado:
        print(f" IDENTIFICADO como: {resultado.nombre} (confianza {resultado.similitud:.1%})")

        vivacidad = ap.calcular_puntaje_vivacidad(audio)
        if vivacidad < config.VIVACIDAD_UMBRAL:
            print(
                f"  Aviso informativo: puntaje de vivacidad bajo ({vivacidad:.2f}). "
                "Esta es solo una heurística ligera, no una prueba de que el audio sea "
                "una grabación reproducida — no se bloquea el acceso, pero si tienes dudas, verifica en persona."
            )
    else:
        print("No se reconoce ninguna voz registrada")


# ---------------------------------------------------------------------------
# OPCIÓN 3: Listado de personas registradas
# ---------------------------------------------------------------------------
def listar_personas() -> None:
    personas = db.listar_speakers()
    if not personas:
        print("  No hay personas registradas.")
        return

    # Diagnóstico por persona: cuántas de sus muestras son compatibles con
    # la versión/dimensión VIGENTE del motor. Sin esto, una persona
    # "registrada" podía nunca ser reconocida sin que el usuario entendiera
    # por qué — el filtro de identificar_persona la excluía en silencio.
    print("\nPersonas registradas:")
    for p in personas:
        estado = ""
        if p["necesita_reentrenar"]:
            estado = "    DESACTUALIZADA (re-entrenar antes de poder identificarse)"
        elif p["muestras_compatibles"] < p["total_muestras"]:
            estado = (
                f"    {p['total_muestras'] - p['muestras_compatibles']} "
                "muestra(s) obsoleta(s) mezclada(s) con muestras vigentes"
            )
        print(
            f"  - {p['nombre']}  (muestras={p['total_muestras']}/{config.MAX_MUESTRAS_POR_PERSONA}, "
            f"desde={p['fecha_registro']:%Y-%m-%d %H:%M}){estado}"
        )


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
            print("  Ingresa solo un número entero (sin letras ni decimales).")
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
        print("  No se detectó suficiente señal de voz para analizar el tono.")
        print("   Revisa el micrófono o intenta hablar más fuerte y cerca.")
    print("-" * 50)


# ---------------------------------------------------------------------------
# OPCIÓN 6 (NUEVA): GESTIONAR PERSONAS REGISTRADAS (CRUD)
# ---------------------------------------------------------------------------
def _renombrar_persona_flujo() -> None:
    nombre_actual = _pedir_nombre("\nNombre actual de la persona: ")
    if nombre_actual is None:
        return
    nombre_nuevo = _pedir_nombre("Nuevo nombre: ")
    if nombre_nuevo is None:
        return

    resultado = db.renombrar_speaker(nombre_actual, nombre_nuevo)
    if resultado == "ok":
        print(f" '{nombre_actual}' ahora se llama '{nombre_nuevo}'.")
    elif resultado == "no_existe":
        print(f"  No se encontró a '{nombre_actual}'.")
    elif resultado == "duplicado":
        print(f" Ya existe una persona registrada como '{nombre_nuevo}'.")


def _reentrenar_persona_flujo() -> None:
    """
    Graba un juego nuevo de muestras (4 a 25) y, SOLO al completarlo,
    reemplaza atómicamente las anteriores (BD y disco). Si se cancela a
    mitad, la persona conserva sus muestras actuales intactas.
    """
    nombre = _pedir_nombre("\nNombre de la persona a re-entrenar: ")
    if nombre is None:
        return
    speaker = db.obtener_speaker(nombre)
    if speaker is None:
        print(f"  No se encontró a '{nombre}'.")
        return

    minimo, maximo = registro.limites_nuevas_muestras("reentrenar")
    confirmar = input(
        f"Se grabarán de {minimo} a {maximo} muestras nuevas que reemplazarán las "
        f"{len(speaker['muestras'])} actuales de '{speaker['nombre']}' al terminar. ¿Continuar? (s/n): "
    ).strip().lower()
    if confirmar != "s":
        print("Cancelado.")
        return

    audios = _grabar_muestras(minimo, maximo)
    if audios is None:
        print("\n  Re-entrenamiento cancelado: se conservan las muestras anteriores.")
        return
    _guardar_perfil_cli(speaker["nombre"], audios, "reentrenar")


def _ampliar_persona_flujo() -> None:
    """Añade muestras a una persona existente sin superar el máximo por persona."""
    nombre = _pedir_nombre("\nNombre de la persona a la que añadir muestras: ")
    if nombre is None:
        return
    speaker = db.obtener_speaker(nombre)
    if speaker is None:
        print(f"  No se encontró a '{nombre}'.")
        return

    minimo, maximo = registro.limites_nuevas_muestras("ampliar", len(speaker["muestras"]))
    if maximo == 0:
        print(f"  '{speaker['nombre']}' ya tiene el máximo de {config.MAX_MUESTRAS_POR_PERSONA} muestras.")
        return
    print(f"'{speaker['nombre']}' tiene {len(speaker['muestras'])} muestra(s); puedes añadir hasta {maximo}.")

    audios = _grabar_muestras(minimo, maximo)
    if not audios:
        print("Cancelado.")
        return
    _guardar_perfil_cli(speaker["nombre"], audios, "ampliar")


def _eliminar_persona_flujo() -> None:
    """
    Elimina a una persona por completo: de la base de datos Y de los
    archivos de audio en disco (ver database.eliminar_speaker_completo).
    Pide confirmación escribiendo el nombre de nuevo.
    """
    nombre = _pedir_nombre("\nNombre exacto de la persona a eliminar: ")
    if nombre is None:
        return

    # Se normaliza igual que el nombre (espacios, NFC) para que "Ana  María" confirme a "Ana María".
    try:
        confirmacion = security.validar_nombre_persona(input(f"Escribe '{nombre}' de nuevo para confirmar el borrado: "))
    except ValueError:
        confirmacion = None
    if confirmacion != nombre:
        print(" Cancelado (el nombre no coincidió).")
        return

    existia, archivos_borrados = db.eliminar_speaker_completo(nombre)

    if existia:
        print(f" '{nombre}' eliminado de la base de datos ({archivos_borrados} archivo(s) borrado(s) de disco).")
    else:
        print(f"  No se encontró a '{nombre}' en el sistema.")


def _reproducir_audios_flujo() -> None:
    """
    Permite elegir a una persona registrada y reproducir, una por una, sus
    muestras de audio guardadas (desencriptándolas en memoria desde
    disco), para verificar físicamente que se grabaron completas y
    correctamente.
    """
    personas = db.listar_speakers()
    if not personas:
        print("  No hay personas registradas.")
        return

    print("\nPersonas registradas:")
    for i, p in enumerate(personas, start=1):
        print(f"  {i}) {p['nombre']} ({p['total_muestras']} muestra(s))")

    seleccion = input("\nNúmero de la persona a escuchar (ENTER para cancelar): ").strip()
    if not seleccion.isdigit() or not (1 <= int(seleccion) <= len(personas)):
        print("Cancelado.")
        return

    speaker = db.obtener_speaker(personas[int(seleccion) - 1]["nombre"])
    if speaker is None or not speaker["muestras"]:
        print("  Esa persona no tiene muestras guardadas.")
        return

    total = len(speaker["muestras"])
    for i, muestra in enumerate(speaker["muestras"], start=1):
        respuesta = input(
            f"\nMuestra {i}/{total} de '{speaker['nombre']}' "
            f"(grabada el {muestra['fecha_creacion']:%Y-%m-%d %H:%M}). "
            "¿Reproducir? (s/n, o 'salir' para terminar): "
        ).strip().lower()
        if respuesta == "salir":
            break
        if respuesta != "s":
            continue
        try:
            audio = ap.cargar_audio_desde_archivo(db.ruta_audio_segura(muestra["archivo_audio"]))
            ap.reproducir_audio(audio)
        except FileNotFoundError:
            print("  El archivo de audio ya no existe en disco.")
        except Exception:
            log.exception("Fallo al reproducir una muestra")
            print("  No se pudo reproducir la muestra.")


def gestionar_personas() -> None:
    """Submenú para editar personas ya registradas (renombrar / re-entrenar / escuchar)."""
    while True:
        print("\n" + "-" * 50)
        print("  GESTIONAR PERSONAS REGISTRADAS")
        print("-" * 50)
        print("1) Ver personas registradas")
        print("2) Renombrar una persona")
        print("3) Re-entrenar (reemplazar sus muestras de voz)")
        print(f"4) Añadir muestras (hasta {config.MAX_MUESTRAS_POR_PERSONA} por persona)")
        print("5) Reproducir audios guardados (verificar grabaciones)")
        print("6) Volver al menú principal")
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "1":
            listar_personas()
        elif opcion == "2":
            _renombrar_persona_flujo()
        elif opcion == "3":
            _reentrenar_persona_flujo()
        elif opcion == "4":
            _ampliar_persona_flujo()
        elif opcion == "5":
            _reproducir_audios_flujo()
        elif opcion == "6":
            return
        else:
            print(" Opción no válida, intenta de nuevo.")


NOMBRE_PRUEBA = "_usuario_prueba_"


def probar_base_datos(audio) -> bool:
    """Escribe un perfil de prueba (con el mínimo de muestras), lo relee y lo elimina."""
    db.init_db()
    db.eliminar_speaker_completo(NOMBRE_PRUEBA)  # restos de una prueba interrumpida
    try:
        registro.guardar_perfil(NOMBRE_PRUEBA, [audio] * config.MIN_MUESTRAS_POR_PERSONA, "registro")
        return any(nombre == NOMBRE_PRUEBA for nombre, _, _ in db.obtener_embeddings_todos())
    finally:
        db.eliminar_speaker_completo(NOMBRE_PRUEBA)


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

    El registro '_usuario_prueba_' creado aquí se borra al terminar. Los
    nombres que empiezan por '_' están reservados: el validador de nombres
    no permite que una persona real se llame así.
    """
    print("\n" + "=" * 60)
    print(" EJECUTANDO PRUEBAS DE VERIFICACIÓN AUTOMÁTICAS")
    print("=" * 60)

    errores = []
    audio_test = None

    # --- Test 1: Micrófono ---
    print("\n[1/3] Probando captura de micrófono (2 segundos)...")
    try:
        audio_test = ap.grabar_audio(duracion=2)
        volumen = float(np.abs(audio_test).mean())
        if volumen < 1e-6:
            print("     Se grabó audio, pero parece silencio total (revisa el micrófono).")
        else:
            print(f"    Micrófono OK. Nivel de señal promedio: {volumen:.6f}")
    except Exception as e:
        errores.append(f"Micrófono: {e}")
        print(f"   Error al grabar: {e}")
        print("   ↳ Se usará una señal sintética para poder continuar con las demás pruebas.")
        audio_test = (np.random.randn(config.SAMPLE_RATE * 2) * 0.01).astype(np.float32)

    # --- Test 2: Base de datos ---
    print("\n[2/3] Probando conexión, escritura y lectura en la base de datos...")
    try:
        if probar_base_datos(audio_test):
            print("    Base de datos OK. Registro de prueba escrito, releído y eliminado en MongoDB.")
        else:
            raise RuntimeError("El registro de prueba no se encontró después de guardarlo.")
    except Exception:
        log.exception("Prueba de base de datos fallida")
        errores.append("Base de datos: no se pudo escribir/leer (detalles en data/logs/voiceid.log)")
        print("    Error de base de datos (detalles en data/logs/voiceid.log)")

    # --- Test 3: Extracción de embedding y comparación (modelo) ---
    print("\n[3/3] Probando extracción de embedding y comparación...")
    try:
        emb1 = ap.extraer_embedding(audio_test)
        emb2 = ap.extraer_embedding(audio_test)  # misma señal -> similitud esperada ≈ 1.0
        score = ap.similitud_coseno(emb1, emb2)
        if score > 0.99:
            print(f"    Modelo OK. Similitud de la muestra consigo misma: {score:.4f} (esperado ≈ 1.0000)")
        else:
            raise RuntimeError(f"Similitud inesperadamente baja: {score:.4f}")
    except Exception as e:
        errores.append(f"Modelo/embedding: {e}")
        print(f"    Error al procesar el embedding: {e}")

    # --- Resumen final ---
    print("\n" + "=" * 60)
    if not errores:
        print(" TODAS LAS PRUEBAS PASARON CORRECTAMENTE. El sistema está listo para usarse.")
    else:
        print(f"  Se encontraron {len(errores)} problema(s):")
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
            print("Hasta luego ")
            sys.exit(0)
        else:
            print(" Opción no válida, intenta de nuevo.")


def configurar_consola() -> None:
    """
    Si la salida no admite UTF-8 (redirigida a un archivo o consola antigua cp1252),
    los emojis y símbolos como «≈» se sustituyen en vez de tumbar el programa
    con UnicodeEncodeError.
    """
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _arrancar() -> None:
    configurar_consola()
    security.configurar_logs()
    try:
        security.verificar_clave_cifrado()
        ap.verificar_motor()
        db.init_db()
    except (db.BaseDatosError, RuntimeError) as e:
        print(f" {e}")
        sys.exit(1)
    try:
        # Ctrl+C o fin de entrada (Ctrl+Z) en el PIN o en cualquier menú: salida limpia, sin traza.
        if not security.pedir_acceso():
            sys.exit(1)
        print("⚙️  Preparando el motor de voz (una sola vez, esto puede tardar unos segundos)...")
        ap.precalentar_motor()
        menu()
    except (KeyboardInterrupt, EOFError):
        print("\nHasta luego ")
    except db.BaseDatosError as e:
        print(f" {e}")
        sys.exit(1)
    except Exception:
        # Nunca se muestra la traza interna al usuario: queda en el log.
        log.exception("Error inesperado en el CLI")
        print(" Error inesperado. Los detalles técnicos se guardaron en data/logs/voiceid.log")
        sys.exit(1)


if __name__ == "__main__":
    _arrancar()