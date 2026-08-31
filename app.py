# ==============================================================================
# IMPORTACIÓN DE LIBRERÍAS Y MÓDULOS DEL SISTEMA
# ==============================================================================
import streamlit as st                  # Framework principal para la interfaz web interactiva
import pandas as pd                     # Manipulación y análisis de datos en tablas y DataFrames
import requests                         # Realización de peticiones HTTP (ej. comunicación con webhooks de n8n)
import psycopg2                         # Conector oficial para bases de datos relacionales PostgreSQL
import base64                           # Codificación y decodificación de datos binarios en texto base64
import hashlib                          # Generación de funciones hash criptográficas (ej. SHA-256)
import html                             # Utilidades para escapar caracteres especiales en HTML
import io                               # Manejo de flujos de datos en memoria (bytes IO)
import os                               # Interacción con el sistema operativo y variables de entorno
import re                               # Operaciones con expresiones regulares para limpieza de textos
import uuid                             # Generación de identificadores únicos universales (UUIDv4)
from pathlib import Path                # Manejo de rutas de archivos y directorios de forma multiplataforma
from PIL import Image, ImageOps, UnidentifiedImageError  # Procesamiento y manipulación avanzada de imágenes PIL
import numpy as np                      # Computación numérica y manejo de matrices multidimensionales
import cv2                              # Librería OpenCV para procesamiento de imágenes y visión artificial
import json                             # Análisis y serialización de datos estructurados en formato JSON
import torch                            # Framework principal de aprendizaje profundo (Deep Learning)
import torchvision.models as models     # Modelos de visión por computadora preentrenados (ej. ResNet18)
import torchvision.transforms as transforms  # Transformaciones y normalizaciones de tensores para imágenes
from google import genai                # Cliente oficial de Google GenAI para interactuar con Gemini
from google.genai import types          # Tipos y configuraciones de esquemas para la API de Gemini
from dotenv import load_dotenv          # Carga de variables de entorno desde archivos locales .env

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y CONSTANTES GLOBALES
# ==============================================================================
APP_DIR = Path(__file__).resolve().parent          # Obtiene el directorio base de la aplicación actual
ASSETS_DIR = APP_DIR / "assets"                    # Define la ruta absoluta hacia la carpeta de recursos/assets
load_dotenv(APP_DIR / ".env")                      # Carga las credenciales y variables secretas desde el archivo .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")  # Define el modelo de IA de Gemini a utilizar
IDENTIFICATION_ALGORITHM_VERSION = "vector-cascaded-v3"       # Versión actual del algoritmo de identificación
CANTIDAD_CANDIDATOS_1_N = 5                        # Número máximo de candidatos (Top-K) a evaluar en la cascada 1:N

def configurar_gemini():
    """Valida y retorna una instancia configurada del cliente oficial de Google GenAI."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY") # Intenta obtener la API Key desde los secretos de Streamlit
        except (FileNotFoundError, KeyError):
            api_key = None
    if not api_key:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Configúrala en .env, como variable de entorno o en .streamlit/secrets.toml."
        )
    return genai.Client(api_key=api_key)

# ==============================================================================
# 0. CONFIGURACIÓN DEL EXTRACTOR DE EMBEDDINGS (TORCH + OPENCV)
# ==============================================================================
@st.cache_resource
def cargar_modelo_extractor():
    """Carga ResNet18 preentrenada y elimina su capa final para dejarla como extractor de 512 dimensiones."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # Remueve la última capa densa de clasificación
    extractor.eval()  # Pone el modelo en modo de evaluación/inferencia
    return extractor

# Define la cadena de transformaciones estándar que requiere ResNet18 para procesar imágenes
transformacion_embedding = transforms.Compose([
    transforms.Resize((224, 224)),                          # Redimensiona la imagen a un cuadro exacto de 224x224 px
    transforms.Grayscale(num_output_channels=3),            # Convierte la imagen a escala de grises manteniendo 3 canales RGB
    transforms.ToTensor(),                                  # Transforma la imagen PIL a un tensor numérico de PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normaliza con medias y desviaciones estándar de ImageNet
])

def generar_embedding_grafomotor(imagen_pil):
    """Transforma una imagen PIL en un vector numérico de 512 dimensiones compatible con pgvector."""
    extractor = cargar_modelo_extractor()
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)  # Convierte PIL a formato OpenCV (BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)                 # Convierte la imagen a escala de grises
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # Aplica umbralizado inverso para aislar la tinta
    pil_input = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)) # Vuelve a convertir la máscara a imagen PIL RGB
    
    tensor_img = transformacion_embedding(pil_input).unsqueeze(0)   # Aplica transformaciones y añade dimensión de lote (batch)
    with torch.no_grad():
        features = extractor(tensor_img)                            # Pasa la imagen por la red para extraer características profundas
    vector = features.squeeze().numpy().tolist()                    # Extrae el vector resultante de 512 dimensiones y lo convierte a lista
    return vector

# ==============================================================================
# 1. CONFIGURACIÓN GENERAL DE LA PÁGINA Y ESTILOS CSS
# ==============================================================================
st.set_page_config(
    page_title="GraphoID AI - Biometría Caligráfica Multimodal", 
    page_icon="✍️", 
    layout="wide"
)

# Inyección de estilos CSS personalizados y Tailwind CSS para un diseño profesional corporativo
st.markdown("""
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <style>
        [data-testid="stAppViewContainer"] {
            background: #eef3f4;
        }
        [data-testid="stMainBlockContainer"] {
            background: #f8faf9;
            border-left: 1px solid #dce6e8;
            border-right: 1px solid #dce6e8;
            padding-top: 0;
        }
        [data-testid="stExpandSidebarButton"] {
            width: 3.75rem;
            height: 2.5rem;
            gap: 0.1rem;
            border: 1px solid #b9d1d5;
            border-radius: 6px;
            background: #ffffff;
            color: #214f61;
            box-shadow: 0 4px 12px rgba(28, 63, 75, 0.12);
        }
        [data-testid="stExpandSidebarButton"]::before {
            content: "chat";
            font-family: "Material Symbols Rounded";
            font-size: 1.4rem;
            font-weight: normal;
            line-height: 1;
            color: #318fb5;
        }
        [data-testid="stExpandSidebarButton"]:hover {
            border-color: #69a9a5;
            background: #edf7f6;
            color: #17324d;
        }
        h1, h2, h3 { color: #1e3a8a; }
        
        div.stButton > button, 
        div.stDownloadButton > button, 
        button[kind="primary"], 
        button[kind="secondary"],
        [data-testid="stFormSubmitButton"] > button {
            background-color: #1B77EA !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(27, 119, 234, 0.25) !important;
            transition: background-color 0.2s ease, transform 0.1s ease !important;
        }
        div.stButton > button:hover, 
        div.stDownloadButton > button:hover, 
        button[kind="primary"]:hover, 
        button[kind="secondary"]:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background-color: #155fc7 !important;
            color: #ffffff !important;
        }

        [data-testid="stVideo"] {
            aspect-ratio: 16 / 9;
            margin: 0.25rem 0 1.25rem;
            overflow: hidden;
            border: 1px solid #b9d1d5;
            border-radius: 8px;
            background: #102b4a;
            box-shadow: 0 10px 26px rgba(26, 58, 76, 0.13);
        }
        [data-testid="stVideo"] video {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: contain;
        }
        .app-header {
            position: relative;
            width: 100%;
            margin: 0 0 1.75rem;
            overflow: hidden;
            border: 0;
            border-radius: 4px;
            background: #102b4a;
        }
        .app-header::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            box-shadow: inset 0 0 2px 14px #f8faf9;
        }
        .app-header img {
            width: 100%;
            height: auto;
            display: block;
        }
        .feature-card {
            background: #ffffff;
            border: 1px solid #cbdadd;
            border-top: 4px solid #1B77EA;
            border-radius: 8px;
            box-shadow: 0 8px 20px rgba(28, 55, 74, 0.10);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            height: 330px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 30px rgba(28, 55, 74, 0.13);
            border-color: #1B77EA;
        }
        .feature-card-media {
            width: 100%;
            height: 150px;
            flex: 0 0 150px;
            overflow: hidden;
            background: #edf4f5;
            border-bottom: 1px solid #dce6e8;
        }
        .feature-card-media img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center 28%;
            display: block;
        }
        .feature-card-content {
            display: flex;
            flex: 1;
            flex-direction: column;
            padding: 0.9rem 1rem 1rem;
        }
        .feature-card-tag {
            align-self: flex-start;
            margin-bottom: 0.5rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: #eaf1fb;
            color: #1B77EA;
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .feature-card h3 {
            margin: 0 0 0.45rem;
            color: #17324d;
            font-size: 1rem;
            font-weight: 750;
        }
        .feature-card p {
            margin: 0;
            color: #60717e;
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .validation-result {
            margin-top: 1rem;
            padding: 1rem 1.15rem;
            border: 1px solid #c5dadd;
            border-radius: 8px;
            background: #f5faf9;
        }
        [data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid #c9dcdf;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 5px 14px rgba(31, 68, 81, 0.06);
        }
    </style>
""", unsafe_allow_html=True)

def render_feature_card(image_name, tag, title, description):
    """Renderiza una tarjeta visual interactiva para la sección Home."""
    image_bytes = (ASSETS_DIR / image_name).read_bytes()
    image_data = base64.b64encode(image_bytes).decode("ascii")
    st.markdown(
        f"""
        <article class="feature-card">
            <div class="feature-card-media">
                <img src="data:image/jpeg;base64,{image_data}" alt="{title}" loading="lazy">
            </div>
            <div class="feature-card-content">
                <span class="feature-card-tag">{tag}</span>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 2. CABEZOTE TECNOLÓGICO INSTITUCIONAL
# ==============================================================================
header_path = ASSETS_DIR / "Cabezote.jpg"
if header_path.exists():
    header_bytes = header_path.read_bytes()
    header_data = base64.b64encode(header_bytes).decode("ascii")
    st.markdown(
        f"""
        <header class="app-header">
            <img src="data:image/jpeg;base64,{header_data}" alt="GraphoID AI">
        </header>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 3. CONEXIÓN A BASE DE DATOS Y UTILIDADES DE PROCESAMIENTO
# ==============================================================================
def obtener_conexion_db():
    """Establece y retorna una conexión activa con la base de datos PostgreSQL local."""
    return psycopg2.connect(
        host="localhost", 
        database="graphoid", 
        user="postgres", 
        password="123456", 
        port="5432"
    )

@st.cache_data(ttl=60)
def cargar_metricas_db():
    """Consulta métricas globales de la base de datos con caché de 60 segundos."""
    try:
        conn = obtener_conexion_db()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(DISTINCT subject_id) FROM biometric_subjects;")
        total_personas = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM biometric_samples;")
        total_muestras = cur.fetchone()[0]
        
        try:
            cur.execute("SELECT COUNT(*) FROM audit_events WHERE status IN ('AUTHENTICATED', 'Aprobado');")
            exitosos = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM audit_events;")
            total_eventos = cur.fetchone()[0]
            if total_eventos > 0:
                tasa_db = (exitosos / total_eventos) * 100
                tasa_str = f"{tasa_db:.1f}%"
            else:
                tasa_str = "100.0%" if total_muestras > 0 else "Sin registros"
        except Exception:
            tasa_str = "100.0%" if total_muestras > 0 else "Sin registros"

        cur.close()
        conn.close()
        return total_personas, total_muestras, tasa_str
    except Exception:
        return 0, 0, "Sin registros"


def preparar_imagen_manuscrito(origen):
    """Carga, corrige orientación EXIF, valida dimensiones mínimas y normaliza la imagen a RGB."""
    try:
        if hasattr(origen, "getvalue"):
            origen = io.BytesIO(origen.getvalue())
        with Image.open(origen) as imagen:
            imagen.load()
            imagen = ImageOps.exif_transpose(imagen) # Corrige la rotación automática basada en metadatos EXIF
            if imagen.width < 100 or imagen.height < 100:
                raise ValueError("La imagen debe medir al menos 100 x 100 píxeles.")
            imagen = imagen.convert("RGB")
            imagen.thumbnail((2048, 2048), Image.Resampling.LANCZOS) # Redimensiona manteniendo proporción si es muy grande
            return imagen.copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("El archivo no es una imagen PNG o JPEG válida.") from error


def evaluar_calidad_tecnica(imagen):
    """Evalúa de forma automática la nitidez, iluminación, contraste y distribución de tinta de la imagen."""
    gris = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    alto, ancho = gris.shape
    total_pixeles = alto * ancho

    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())    # Calcula la varianza del Laplaciano como métrica de enfoque
    brillo = float(gris.mean())                               # Calcula el nivel promedio de iluminación (0-255)
    contraste = float(gris.std())                             # Calcula la desviación estándar como medida de contraste
    _, mascara_tinta = cv2.threshold(
        gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    pixeles_tinta = int(np.count_nonzero(mascara_tinta))
    proporcion_tinta = pixeles_tinta / total_pixeles

    # Evalúa la concentración de tinta en los bordes para detectar malos recortes o encuadres deficientes
    borde_y = max(1, int(alto * 0.03))
    borde_x = max(1, int(ancho * 0.03))
    mascara_borde = np.zeros_like(mascara_tinta)
    mascara_borde[:borde_y, :] = 255
    mascara_borde[-borde_y:, :] = 255
    mascara_borde[:, :borde_x] = 255
    mascara_borde[:, -borde_x:] = 255
    tinta_en_borde = np.count_nonzero(cv2.bitwise_and(mascara_tinta, mascara_borde))
    proporcion_borde = tinta_en_borde / max(1, pixeles_tinta)

    criterios = [
        {
            "criterio": "Resolución",
            "cumple": ancho >= 300 and alto >= 250 and total_pixeles >= 120_000,
            "detalle": f"{ancho} x {alto} px (mínimo 300 x 250 y 120.000 px totales)",
        },
        {
            "criterio": "Nitidez",
            "cumple": nitidez >= 80,
            "detalle": f"Índice {nitidez:.1f} (mínimo 80)",
        },
        {
            "criterio": "Iluminación",
            "cumple": 70 <= brillo <= 249.9,
            "detalle": f"Nivel {brillo:.1f}/255 (rango permitido 70-249.9)",
        },
        {
            "criterio": "Contraste",
            "cumple": contraste >= 20,
            "detalle": f"Índice {contraste:.1f} (mínimo 20)",
        },
        {
            "criterio": "Cantidad de trazos",
            "cumple": 0.01 <= proporcion_tinta <= 0.35,
            "detalle": f"{proporcion_tinta * 100:.1f}% del área (rango 1%-35%)",
        },
        {
            "criterio": "Encuadre y recorte",
            "cumple": proporcion_borde <= 0.12,
            "detalle": f"{proporcion_borde * 100:.1f}% de tinta en bordes (máximo 12%)",
        },
    ]

    return {
        "apta": all(criterio["cumple"] for criterio in criterios),
        "criterios": criterios,
        "motivos": [
            criterio["criterio"]
            for criterio in criterios
            if not criterio["cumple"]
        ],
    }


def mostrar_evaluacion_calidad(resultado, titulo="Control técnico de la imagen"):
    """Muestra un reporte desplegable en Streamlit con los resultados de la evaluación técnica."""
    with st.expander(titulo, expanded=not resultado["apta"]):
        if resultado["apta"]:
            st.success(
                "La imagen cumple todos los criterios técnicos.",
                icon=":material/check_circle:",
            )
        else:
            st.error(
                "La imagen no es apta. Corrige los criterios marcados antes de continuar.",
                icon=":material/cancel:",
            )
        filas = [
            {
                "Estado": "Aprobado" if criterio["cumple"] else "Revisar",
                "Criterio": criterio["criterio"],
                "Medición": criterio["detalle"],
            }
            for criterio in resultado["criterios"]
        ]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


def cargar_referencias_db():
    """Carga las imágenes de referencia base de todos los usuarios registrados desde la base de datos."""
    referencias = {}
    advertencias = []
    with obtener_conexion_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (subject_id)
                    subject_id,
                    feature_vector ->> 'ruta_base' AS ruta_base
                FROM biometric_samples
                WHERE feature_vector ->> 'ruta_base' IS NOT NULL
                ORDER BY subject_id, created_at DESC, sample_id DESC;
            """)
            muestras = cur.fetchall()

    for sujeto, ruta_guardada in muestras:
        ruta_imagen = Path(ruta_guardada)
        if not ruta_imagen.is_absolute():
            ruta_imagen = APP_DIR / ruta_imagen
        try:
            if not ruta_imagen.is_file():
                raise FileNotFoundError(ruta_imagen)
            imagen = preparar_imagen_manuscrito(ruta_imagen)
            evaluacion = evaluar_calidad_tecnica(imagen)
            if not evaluacion["apta"]:
                fallos = ", ".join(evaluacion["motivos"])
                advertencias.append(f"{sujeto}: calidad insuficiente ({fallos})")
                continue
            referencias[sujeto] = imagen
        except (ValueError, OSError) as error:
            advertencias.append(f"{sujeto}: {error}")

    return referencias, advertencias


def extraer_json_respuesta(respuesta):
    """Extrae y parsea de manera segura un objeto JSON desde la respuesta de texto de Gemini."""
    texto = getattr(respuesta, "text", "") or ""
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip(), flags=re.IGNORECASE)
    try:
        resultado = json.loads(texto)
    except json.JSONDecodeError as error:
        raise RuntimeError("Gemini devolvió una respuesta que não es JSON válido.") from error
    if not isinstance(resultado, dict):
        raise RuntimeError("Gemini devolvió un resultado con formato inesperado.")
    return resultado


def normalizar_puntaje(valor):
    """Asegura que un puntaje numérico se encuentre estrictamente en el rango decimal de 0.0 a 1.0."""
    try:
        return min(1.0, max(0.0, float(valor)))
    except (TypeError, ValueError):
        return 0.0


def mensaje_error_validacion(error):
    """Traduce errores técnicos de la API en mensajes claros y amigables para el usuario final."""
    detalle = str(error).lower()
    if "503" in detalle or "unavailable" in detalle or "high demand" in detalle:
        return (
            "El servicio de análisis está temporalmente ocupado por alta demanda. "
            "Tu muestra no presenta ningún problema; espera unos minutos y vuelve a intentarlo."
        )
    if "429" in detalle or "resource_exhausted" in detalle or "quota" in detalle:
        return "Se alcanzó el límite temporal de validaciones. Espera unos minutos antes de volver a intentarlo."
    if "timeout" in detalle or "connection" in detalle:
        return "No fue posible conectarse con el servicio de análisis. Verifica tu conexión e inténtalo nuevamente."
    return "No pudimos completar la validación en este momento. Inténtalo nuevamente más tarde."


def imagen_a_parte_gemini(imagen):
    """Convierte una imagen PIL en un objeto de tipo Part optimizado para ser enviado a la API de Gemini."""
    buffer = io.BytesIO()
    imagen.save(buffer, format="JPEG", quality=95, optimize=True)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def normalizar_trazos_para_cotejo(imagen):
    """Aplica normalización geométrica avanzada y eliminación de ruido con OpenCV sobre el trazo."""
    gris = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    dimension_maxima = max(gris.shape)
    sigma = max(5.0, dimension_maxima / 35)
    fondo = cv2.GaussianBlur(gris, (0, 0), sigmaX=sigma, sigmaY=sigma) # Estima la iluminación de fondo mediante desenfoque gaussiano
    fondo = np.maximum(fondo, 1)
    gris_uniforme = cv2.divide(gris, fondo, scale=255)                # Normaliza la iluminación eliminando sombras de la hoja

    _, tinta = cv2.threshold(
        gris_uniforme, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(tinta, 8)
    tinta_limpia = np.zeros_like(tinta)
    area_minima = max(4, round(tinta.size * 0.000004))                # Define área mínima para considerar un trazo válido y descartar ruido
    for etiqueta in range(1, cantidad):
        if estadisticas[etiqueta, cv2.CC_STAT_AREA] >= area_minima:
            tinta_limpia[etiquetas == etiqueta] = 255

    puntos = cv2.findNonZero(tinta_limpia)
    if puntos is None:
        raise ValueError("No se detectaron trazos utilizables para el cotejo.")

    x, y, ancho, alto = cv2.boundingRect(puntos)                      # Calcula las coordenadas del rectángulo delimitador del trazo
    margen_x = max(12, round(ancho * 0.06))
    margen_y = max(12, round(alto * 0.10))
    lienzo = np.full(
        (alto + 2 * margen_y, ancho + 2 * margen_x), 255, dtype=np.uint8
    )
    recorte = 255 - tinta_limpia[y:y + alto, x:x + ancho]
    lienzo[margen_y:margen_y + alto, margen_x:margen_x + ancho] = recorte

    imagen_normalizada = Image.fromarray(lienzo, mode="L").convert("RGB")
    imagen_normalizada.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    return imagen_normalizada


def calcular_diagnostico_comparacion(imagen_consulta, imagen_referencia):
    """Calcula métricas de diagnóstico visual y huellas SHA-256 para control de integridad entre imágenes."""
    consulta_norm = normalizar_trazos_para_cotejo(imagen_consulta)
    referencia_norm = normalizar_trazos_para_cotejo(imagen_referencia)

    consulta_gris = cv2.cvtColor(np.array(consulta_norm), cv2.COLOR_RGB2GRAY)
    referencia_gris = cv2.cvtColor(np.array(referencia_norm), cv2.COLOR_RGB2GRAY)
    consulta_muestra = cv2.resize(consulta_gris, (512, 512), interpolation=cv2.INTER_AREA)
    referencia_muestra = cv2.resize(referencia_gris, (512, 512), interpolation=cv2.INTER_AREA)

    consulta_tinta = consulta_muestra < 200
    referencia_tinta = referencia_muestra < 200
    coincidencia_pixeles = float(np.mean(consulta_tinta == referencia_tinta))
    proporcion_tinta_consulta = float(np.mean(consulta_tinta))
    proporcion_tinta_referencia = float(np.mean(referencia_tinta))

    buffer_consulta = io.BytesIO()
    buffer_referencia = io.BytesIO()
    consulta_norm.save(buffer_consulta, format="PNG")
    referencia_norm.save(buffer_referencia, format="PNG")
    hash_consulta = hashlib.sha256(buffer_consulta.getvalue()).hexdigest()
    hash_referencia = hashlib.sha256(buffer_referencia.getvalue()).hexdigest()

    return {
        "consulta_normalizada": consulta_norm,
        "referencia_normalizada": referencia_norm,
        "imagen_exacta": hash_consulta == hash_referencia,
        "coincidencia_pixeles": coincidencia_pixeles,
        "tinta_consulta": proporcion_tinta_consulta,
        "tinta_referencia": proporcion_tinta_referencia,
        "dimensiones_consulta": f"{imagen_consulta.width} x {imagen_consulta.height} px",
        "dimensiones_referencia": f"{imagen_referencia.width} x {imagen_referencia.height} px",
    }


def mostrar_diagnostico_comparacion(
    imagen_consulta,
    imagen_referencia,
    sujeto_referencia,
    titulo="Comparación con la referencia base",
):
    """Muestra en la interfaz un panel comparativo con los trazos normalizados y métricas analíticas."""
    diagnostico = calcular_diagnostico_comparacion(imagen_consulta, imagen_referencia)
    with st.expander(titulo, expanded=True):
        if diagnostico["imagen_exacta"]:
            st.warning(
                "La muestra cargada es visualmente idéntica a la referencia base normalizada. "
                "Realiza otra captura o escribe una muestra nueva antes de validar. "
                "Este control de integridad no determina por sí solo la autoría.",
                icon=":material/content_copy:",
            )
        else:
            st.success(
                "La muestra es independiente de la referencia base. La identificación se basa "
                "en rasgos grafomotores, aunque el texto sea diferente.",
                icon=":material/difference:",
            )

        columna_consulta, columna_referencia = st.columns(2)
        with columna_consulta:
            st.image(
                diagnostico["consulta_normalizada"],
                caption="Muestra consultada (trazos normalizados)",
                width="stretch",
            )
        with columna_referencia:
            st.image(
                diagnostico["referencia_normalizada"],
                caption=f"Referencia más próxima: {sujeto_referencia}",
                width="stretch",
            )

        medidas = pd.DataFrame([
            {
                "Medida": "Dimensiones originales",
                "Muestra consultada": diagnostico["dimensiones_consulta"],
                "Referencia base": diagnostico["dimensiones_referencia"],
            },
            {
                "Medida": "Área ocupada por tinta",
                "Muestra consultada": f"{diagnostico['tinta_consulta'] * 100:.1f}%",
                "Referencia base": f"{diagnostico['tinta_referencia'] * 100:.1f}%",
            },
            {
                "Medida": "Coincidencia de píxeles normalizados",
                "Muestra consultada": f"{diagnostico['coincidencia_pixeles'] * 100:.1f}%",
                "Referencia base": "100% solo si es copia exacta",
            },
        ])
        st.dataframe(medidas, hide_index=True, width="stretch")

# ==============================================================================
# 4. MOTOR DE VALIDACIÓN Y PROMPTS FORENSES (1 a 1 y 1 a N en Cascada)
# ==============================================================================
PROMPT_VALIDAR_MANUSCRITO = """
Actúa como un Perito Calígrafo Forense y experto en Visión Artificial.
Analiza la imagen proporcionada. Tu único objetivo es determinar si la imagen contiene **escritura a mano auténtica (manuscrita)** hecha con bolígrafo, lápiz, pluma o marcador sobre papel/superficie.

REGLAS ESTRICTAS DE RECHAZO:
- Si la imagen muestra un paisaje, una persona, un objeto, un animal, un gráfico generado por computadora, una fotografía de un objeto cotidiano, memes o texto estrictamente impreso en tipografía digital sin trazos de escritura manual humana, debes rechazarla.

Devuelve ÚNICAMENTE un objeto JSON válido con la siguiente estructura:
{
  "is_handwriting": <booleano true si es escritura manuscrita real humana, false para cualquier otro tipo de imagen>,
  "reason": "<breve explicación pericial de por qué es o no es un manuscrito>"
}
"""

PROMPT_IDENTIFICACION_GLOBAL = """
Actúa como Perito Calígrafo Forense y experto en Grafotecnia. Compara la MUESTRA DE CONSULTA 
contra los candidatos seleccionados por similitud vectorial.

REGLAS OBLIGATORIAS:
1. Identifica al autor, no una copia de la imagen ni la coincidencia literal del texto.
2. Las muestras pueden contener textos, palabras, extensiones y distribuciones diferentes. No penalices esas diferencias ni exijas coincidencia de píxeles, renglones o contenido.
3. Para cada candidato, compara las letras y combinaciones homólogas que sí estén presentes en ambas muestras. Complementa el cotejo con rasgos independientes del contenido: construcción y proporción de grafías, inclinación, enlaces, espaciado, línea base, presión aparente, ritmo y continuidad.
4. Basa las discrepancias únicamente en rasgos grafomotores comparables. La ausencia de una letra en una muestra no constituye una discrepancia.
5. Puntúa cada candidato entre 0.0 y 1.0 según la probabilidad de autoría común. El puntaje no representa parecido visual global, contenido compartido ni copia exacta.
6. Devuelve un elemento en rankings para cada candidato suministrado y usa exactamente su etiqueta.
7. Devuelve is_match=true solo cuando existan varios rasgos individualizantes concordantes y no haya contradicciones grafomotoras relevantes.
8. No fuerces una identidad. Si faltan grafías comparables o hay duda, devuelve is_match=false y best_match_subject="Desconocido".

Devuelve ÚNICAMENTE un objeto JSON válido con este formato:
{
    "is_match": <booleano>,
    "best_match_subject": "<etiqueta exacta o Desconocido>",
    "confidence": <flotante entre 0.0 y 1.0>,
    "analysis_details": "<explicación breve>",
    "rankings": [
        {"subject_id": "<etiqueta exacta>", "similarity_score": <flotante 0.0-1.0>}
    ]
}
"""

PROMPT_VERIFICACION_1_A_1 = """
Actúa como Perito Calígrafo Forense. Compara a ciegas la MUESTRA A con la MUESTRA B.
Determina si ambas pertenecen inequívocamente al mismo autor basándote solo en rasgos grafomotores.

REGLAS DE SEGURIDAD:
1. Trata la comparación como un intento potencial de suplantación y prioriza evitar falsos positivos.
2. Las muestras pueden contener textos, palabras, extensiones y distribuciones diferentes. No exijas que el contenido sea igual ni penalices letras o palabras que solo aparezcan en una muestra.
3. Compara las mismas letras y combinaciones que sí estén presentes en ambas muestras. Complementa el cotejo con rasgos independientes del contenido: inclinación, proporción, enlaces, espaciado, línea base, presión aparente, ritmo y continuidad.
4. El parecido del contenido o del instrumento no demuestra identidad; tampoco la diferencia de contenido demuestra autores distintos.
5. Busca contradicciones grafomotoras únicamente entre rasgos comparables y explica cuáles grafías homólogas sustentan la decisión.
6. Devuelve is_match=true únicamente si no hay discrepancias relevantes y existen varios rasgos individualizantes coincidentes.
7. Ante pocas grafías comparables, ambigüedad o duda, devuelve is_match=false.

Devuelve ÚNICAMENTE un objeto JSON válido con este formato:
{
    "is_match": <booleano true si es el mismo autor, false si hay discrepancias o dudas>,
    "similarity_score": <flotante entre 0.0 y 1.0>,
    "confidence": <flotante entre 0.0 y 1.0>,
    "analysis_details": "<explicación pericial detallada de la coincidencia o rechazo>"
}
"""

def verificar_si_es_manuscrito(imagen_pil):
    """Envía la imagen a Gemini para verificar si contiene escritura manuscrita auténtica."""
    cliente = configurar_gemini()
    try:
        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=[PROMPT_VALIDAR_MANUSCRITO, imagen_a_parte_gemini(imagen_pil)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        resultado = extraer_json_respuesta(respuesta)
        es_manuscrito = resultado.get("is_handwriting") is True
        motivo = str(resultado.get("reason") or "Sin explicación disponible.")
        return es_manuscrito, motivo
    except Exception as e:
        raise RuntimeError(f"No fue posible validar la imagen con Gemini: {e}") from e

def identificar_manuscrito_en_bd_vectorial(imagen_consulta_pil, referencias_usuarios):
    """Ejecuta la arquitectura en cascada: filtro vectorial con pgvector + peritaje profundo en Gemini (1 a N)."""
    # Paso 1: Generar embedding de la consulta mediante ResNet18
    vector_consulta = generar_embedding_grafomotor(imagen_consulta_pil)
    vector_str = "[" + ",".join(map(str, vector_consulta)) + "]"

    # Paso 2: Filtrado rápido en base de datos usando pgvector y operadores de distancia coseno (<=>)
    candidatos_top = []
    with obtener_conexion_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT subject_id, MIN(embedding <=> %s::vector) AS distancia
                FROM biometric_samples
                WHERE embedding IS NOT NULL
                GROUP BY subject_id
                ORDER BY distancia ASC
                LIMIT %s;
            """, (vector_str, CANTIDAD_CANDIDATOS_1_N))
            candidatos_top = cur.fetchall()

    if not candidatos_top:
        return None

    refs_filtradas = {sujeto: referencias_usuarios[sujeto] for sujeto, _ in candidatos_top if sujeto in referencias_usuarios}
    if not refs_filtradas:
        refs_filtradas = referencias_usuarios

    # Paso 3: Validación pericial profunda con Gemini sobre el grupo acotado de candidatos
    cliente = configurar_gemini()
    consulta_normalizada = normalizar_trazos_para_cotejo(imagen_consulta_pil)
    contenido_peticion = [
        PROMPT_IDENTIFICACION_GLOBAL,
        f"Evalúa y devuelve un elemento en rankings para cada uno de los {len(refs_filtradas)} candidatos proporcionados.",
        "MUESTRA DE CONSULTA NORMALIZADA (AUTOR DESCONOCIDO):",
        imagen_a_parte_gemini(consulta_normalizada),
    ]
    
    for indice, (sujeto, img_ref_pil) in enumerate(refs_filtradas.items(), start=1):
        referencia_normalizada = normalizar_trazos_para_cotejo(img_ref_pil)
        contenido_peticion.extend([
            f"CANDIDATO VECTORIAL {indice} - ETIQUETA EXACTA: {sujeto}",
            imagen_a_parte_gemini(referencia_normalizada),
        ])

    try:
        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=contenido_peticion,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        resultado = extraer_json_respuesta(respuesta)
    except Exception as error:
        raise RuntimeError(f"Falló la comparación global en cascada: {error}") from error

    etiquetas_validas = set(refs_filtradas.keys())
    puntajes = {}
    for candidato in resultado.get("rankings", []):
        sujeto = str(candidato.get("subject_id", ""))
        if sujeto in etiquetas_validas:
            puntajes[sujeto] = normalizar_puntaje(candidato.get("similarity_score"))

    if not puntajes:
        return {
            "best_match_subject": "Desconocido",
            "similarity_score": 0.0,
            "confidence": 0.0,
            "vector_similarity": 0.0,
            "matched": False,
            "rankings": [],
            "forensic_analysis": "No se encontraron puntajes válidos en el filtrado vectorial."
        }

    ranking = sorted(puntajes.items(), key=lambda item: item[1], reverse=True)
    mejor_sujeto, mejor_puntaje = ranking[0]
    segundo_puntaje = ranking[1][1] if len(ranking) > 1 else 0.0
    margen = mejor_puntaje - segundo_puntaje
    confianza = normalizar_puntaje(resultado.get("confidence"))
    etiqueta_modelo = str(resultado.get("best_match_subject", "Desconocido"))
    distancias_vectoriales = {sujeto: float(distancia) for sujeto, distancia in candidatos_top}
    compatibilidad_tecnica = normalizar_puntaje(
        1.0 - distancias_vectoriales.get(mejor_sujeto, 1.0)
    )
    
    coincide = (
        resultado.get("is_match") is True
        and etiqueta_modelo == mejor_sujeto
        and mejor_puntaje >= 0.82
        and confianza >= 0.75
        and margen >= 0.10
    )
    detalles = str(resultado.get("analysis_details") or "Sin detalles del cotejo.")

    if coincide:
        return {
            "best_match_subject": mejor_sujeto,
            "similarity_score": mejor_puntaje,
            "confidence": confianza,
            "vector_similarity": compatibilidad_tecnica,
            "matched": True,
            "rankings": ranking,
            "forensic_analysis": f"Identificación en cascada positiva con {mejor_sujeto}. Margen: {margen * 100:.1f}%. {detalles}",
        }

    return {
        "best_match_subject": "Desconocido",
        "similarity_score": mejor_puntaje,
        "confidence": confianza,
        "vector_similarity": compatibilidad_tecnica,
        "matched": False,
        "rankings": ranking,
        "forensic_analysis": f"Resultado no concluyente. Mejor candidato vectorial/pericial: {mejor_sujeto} ({mejor_puntaje * 100:.1f}%), margen: {margen * 100:.1f}%. {detalles}",
    }


def verificar_consistencia_vectorial_1_a_1(imagen_consulta_pil, sujeto_objetivo):
    """Calcula la distancia vectorial orientativa contra la referencia específica de un usuario en la BD."""
    vector_consulta = generar_embedding_grafomotor(imagen_consulta_pil)
    vector_str = "[" + ",".join(map(str, vector_consulta)) + "]"

    with obtener_conexion_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MIN(embedding <=> %s::vector) AS distancia
                FROM biometric_samples
                WHERE subject_id = %s AND embedding IS NOT NULL;
            """, (vector_str, sujeto_objetivo))
            fila = cur.fetchone()

    if not fila or fila[0] is None:
        return {
            "confirmed": False,
            "similarity": 0.0,
            "reason": f"{sujeto_objetivo} no tiene una huella vectorial registrada.",
        }

    distancia_objetivo = float(fila[0])
    similitud = normalizar_puntaje(1.0 - distancia_objetivo)
    confirmado = similitud >= 0.70

    razon = (
        f"Compatibilidad técnica orientativa con la referencia de {sujeto_objetivo}: "
        f"{similitud * 100:.1f}%. Esta medida no interviene en la decisión de identidad."
    )

    return {
        "confirmed": confirmado,
        "similarity": similitud,
        "reason": razon,
    }


def verificar_autenticacion_1_a_1(imagen_consulta_pil, sujeto_objetivo, img_referencia_pil):
    """Ejecuta la validación biométrica estricta de 1 a 1 comparando la muestra contra el usuario indicado."""
    cliente = configurar_gemini()
    consulta_norm = normalizar_trazos_para_cotejo(imagen_consulta_pil)
    referencia_norm = normalizar_trazos_para_cotejo(img_referencia_pil)

    contenido_peticion = [
        PROMPT_VERIFICACION_1_A_1,
        "MUESTRA A:",
        imagen_a_parte_gemini(referencia_norm),
        "MUESTRA B:",
        imagen_a_parte_gemini(consulta_norm),
    ]

    try:
        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=contenido_peticion,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        resultado = extraer_json_respuesta(respuesta)
        score = normalizar_puntaje(resultado.get("similarity_score"))
        confianza = normalizar_puntaje(resultado.get("confidence"))
        verificacion_vectorial = verificar_consistencia_vectorial_1_a_1(
            imagen_consulta_pil,
            sujeto_objetivo,
        )
        matched = (
            resultado.get("is_match") is True
            and score >= 0.90
            and confianza >= 0.90
        )
        detalles = str(resultado.get("analysis_details") or "Sin análisis detallado.")
        razon_vectorial = verificacion_vectorial["reason"]

        return {
            "matched": matched,
            "best_match_subject": sujeto_objetivo if matched else "Desconocido",
            "similarity_score": score,
            "confidence": confianza,
            "vector_similarity": verificacion_vectorial["similarity"],
            "forensic_analysis": f"Verificación 1 a 1 para {sujeto_objetivo}: {'Aprobado' if matched else 'Rechazado'}. {razon_vectorial} Análisis visual: {detalles}"
        }
    except Exception as error:
        raise RuntimeError(f"Falló la verificación 1 a 1: {error}") from error


def generar_certificado_html(resultado):
    """Genera una plantilla HTML profesional y descargable con el dictamen de validación y certificación."""
    estado_color = "#277a65" if resultado["matched"] else "#a14343"
    estado_fondo = "#eaf6f1" if resultado["matched"] else "#fff1f1"
    sujeto = html.escape(str(resultado["best_match_subject"]))
    estado = html.escape(resultado["estado"])
    fecha = html.escape(resultado["fecha"])
    certificado_id = html.escape(resultado["certificado_id"])
    analisis = html.escape(resultado.get("forensic_analysis", ""))

    return f"""<!doctype html>
<html lang="es">
<head><meta charset="utf-8"><title>Certificación {certificado_id}</title></head>
<body style="font-family: Georgia, serif; padding: 40px; background: #edf3f4;">
    <div style="max-width: 800px; margin: auto; padding: 40px; background: #fff; border: 1px solid #9fc4ca;">
        <h2 style="color: #237f9f;">GraphoID AI - Certificado de Validación Biométrica</h2>
        <div style="padding: 10px; background: {estado_fondo}; color: {estado_color}; font-weight: bold;">{estado}</div>
        <p><b>Propietario / Usuario:</b> {sujeto}</p>
        <p><b>Puntaje de Similitud:</b> {resultado["similarity_score"] * 100:.1f}%</p>
        <p><b>Dictamen Forense:</b> {analisis}</p>
        <p><b>Fecha:</b> {fecha}</p>
        <p><b>Código de Verificación:</b> {certificado_id}</p>
    </div>
</body>
</html>"""

# ==============================================================================
# 5. INTEGRACIÓN CON n8n WEBHOOK (AUDITORÍA Y TRAZABILIDAD)
# ==============================================================================
def registrar_en_n8n(subject_id, status, confidence, analisis):
    """Envía los eventos de auditoría biométrica al flujo de automatización en n8n mediante Webhook HTTP POST."""
    url = "http://localhost:5678/webhook/audit-event"
    token_n8n = os.getenv("GRAPHOID_SECURE_TOKEN_2O26", "").strip()
    if not token_n8n:
        raise RuntimeError("Falta GRAPHOID_SECURE_TOKEN_2O26 en el archivo .env.")
    headers = {
        "Authorization": f"Bearer {token_n8n}",
        "Content-Type": "application/json"
    }
    client_ip = "127.0.0.1"
    
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers") and st.context.headers:
            headers_dict = st.context.headers
            client_ip = headers_dict.get("X-Forwarded-For") or headers_dict.get("X-Real-IP") or "127.0.0.1"
            if "," in client_ip:
                client_ip = client_ip.split(",")[0].strip()
    except Exception:
        pass

    try:
        conf_float = float(confidence)
    except (TypeError, ValueError):
        conf_float = 0.0

    payload = {
        "event_id": f"GID-EVT-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex[:8].upper()}",
        "user_id": str(subject_id), 
        "status": str(status), 
        "confidence_score": conf_float,
        "analysis": str(analisis),
        "source_type": "handwriting_verification",
        "ip_address": str(client_ip),
        "device_info": "Streamlit-Multimodal-App",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try: 
        print(f"--- ENVIANDO A n8n --- URL: {url} | Payload: {payload}")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"--- RESPUESTA DE n8n --- Código: {response.status_code} | Cuerpo: {response.text}")
        response.raise_for_status()
        st.session_state["n8n_status"] = "🟢 Activo (Enviado a n8n)"
    except requests.exceptions.RequestException as e:
        error_detalle = e.response.text if e.response is not None else str(e)
        print(f"--- ERROR CRÍTICO EN n8n --- {error_detalle}")
        st.warning(
            "El análisis biométrico finalizó correctamente, pero el evento no pudo "
            "registrarse en el servicio de auditoría. Inténtalo nuevamente más tarde.",
            icon=":material/cloud_off:",
        )
        st.session_state["n8n_status"] = "🟠 Auditoría temporalmente no disponible"

    if "audit_logs" not in st.session_state:
        st.session_state["audit_logs"] = []
    st.session_state["audit_logs"].insert(0, {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Usuario Identificado": str(subject_id),
        "Estado": str(status),
        "Confianza": conf_float
    })

# ==============================================================================
# 6. BARRA LATERAL (ASISTENTE Y ESTADO DE n8n)
# ==============================================================================
with st.sidebar:
    st.subheader("💬 Asistente GraphoID")
    st.write("Soporte analítico impulsado por IA.")
    st.iframe(
        "https://cdn.botpress.cloud/webchat/v3.7/shareable.html?configUrl=https://files.bpcontent.cloud/2026/08/09/22/20260809225145-KARNRHO1.json",
        height=450,
    )
    st.divider()
    st.subheader("⚡ Monitoreo de n8n")
    st.markdown(f"**Estado:** {st.session_state.get('n8n_status', '🟢 En espera')}")

# ==============================================================================
# 7. INTERFAZ PRINCIPAL (PESTAÑAS DE NAVEGACIÓN)
# ==============================================================================
tab_empresa, tab_home, tab_dashboard, tab_registrar, tab_validar_1_1, tab_validar_1_n = st.tabs([
    "🏢 Empresa", "🏠 Home", "📊 Dashboard", "📝 Registrar Formato Base", "🔐 Validación 1 a 1", "🔍 Identificación Global (1 a N)"
])

# --- PESTAÑA EMPRESA ---
with tab_empresa:
    col_logo, col_info = st.columns([1, 2], gap="large")
    with col_logo:
        logo_path = ASSETS_DIR / "logo.png"
        if not logo_path.exists():
            logo_path = ASSETS_DIR / "logo.jpg"
        if logo_path.exists():
            image_bytes = logo_path.read_bytes()
            image_data = base64.b64encode(image_bytes).decode("ascii")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; align-items: center; background: #ffffff; border: 1px solid #cbdadd; border-radius: 8px; padding: 10px; box-shadow: 0 4px 12px rgba(28, 55, 74, 0.05);">
                    <img src="data:image/jpeg;base64,{image_data}" alt="GraphoID AI Logo" style="max-height: 120px; width: auto; object-fit: contain; display: block; margin: auto;">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Coloca una imagen llamada `logo.png` o `logo.jpg` en la carpeta `assets` para mostrar el logo institucional.")
    with col_info:
        st.markdown("#### *Seguridad documental y biometría caligráfica de nueva generación.*")
        st.write(
            "GraphoID AI es una plataforma avanzada de biometría y peritaje caligráfico "
            "impulsada por inteligencia artificial multimodal y arquitectura vectorial en cascada."
        )

    st.divider()
    
    col_mision, col_vision = st.columns(2, gap="large")
    with col_mision:
        st.markdown("### 🎯 Misión")
        st.write(
            "Garantizar la máxima seguridad documental e integridad de la identidad humana "
            "a través de soluciones innovadoras de biometría caligráfica, combinando la precisión "
            "de la visión artificial y la velocidad de las bases de datos vectoriales para proteger "
            "los procesos corporativos, legales y académicos más exigentes."
        )
    with col_vision:
        st.markdown("### 🔭 Visión")
        st.write(
            "Consolidarnos para el año 2030 como la plataforma líder a nivel global en peritaje "
            "caligráfico automatizado e identificación forense por IA, estableciendo el estándar "
            "industrial en la prevención de fraudes por suplantación de identidad en entornos digitales."
        )

    st.divider()

    st.markdown("### 📌 Objetivos")
    col_obj_gen, col_obj_esp = st.columns(2, gap="large")
    with col_obj_gen:
        st.markdown("#### **Objetivo General**")
        st.write(
            "Desarrollar e implementar un sistema automatizado de autenticación e identificación "
            "de escritura manuscrita y firmas mediante una arquitectura híbrida que combine visión "
            "artificial, aprendizaje profundo y bases de datos vectoriales, garantizando una alta "
            "fiabilidad pericial y prevención de fraudes por suplantación."
        )
    with col_obj_esp:
        st.markdown("#### **Objetivos Específicos**")
        st.markdown(
            """
            * **Preprocesar** y estandarizar las muestras de trazos mediante algoritmos de OpenCV para eliminar ruido y asegurar la consistencia geométrica.
            * **Extraer** huellas grafomotoras profundas de 512 dimensiones utilizando redes neuronales convolucionales (ResNet18).
            * **Optimizar** la búsqueda y el cotejo masivo de identidades (1 a N) implementando bases de datos vectoriales (`pgvector`).
            * **Automatizar** el peritaje forense profundo aplicando modelos multimodales (Google Gemini) bajo criterios grafotécnicos estrictos.
            * **Garantizar** la trazabilidad y el registro seguro de eventos mediante la integración de flujos de auditoría con webhooks en n8n.
            """
        )

# --- PESTAÑA HOME ---
with tab_home:
    st.video(str(ASSETS_DIR / "avatar_parlant.mp4"), width="stretch")
    col1, col2, col3, col4 = st.columns(4)
    with col1: render_feature_card("Certificación Documental.jpg", "Integridad", "Certificación", "Respaldo forense.")
    with col2: render_feature_card("Inspección de Trazo.jpg", "Visión IA", "Identificación", "Cotejo global en DB.")
    with col3: render_feature_card("Validaciòn Firma.jpg", "Biometría", "Firma", "Contraste multivariable.")
    with col4: render_feature_card("Validación Texto.jpg", "OCR", "Lectura", "Análisis avanzado.")

# --- PESTAÑA DASHBOARD ---
with tab_dashboard:
    st.markdown("## 📊 Dashboard Analítico y Auditoría Global")
    db_personas, db_muestras, tasa_exito_db = cargar_metricas_db()
    logs = st.session_state.get("audit_logs", [])
    
    if logs:
        total_logs = len(logs)
        exitosos_logs = [log for log in logs if log.get("Estado") in ["AUTHENTICATED", "Aprobado"] or "Identificado" in str(log.get("Estado"))]
        alertas_logs = total_logs - len(exitosos_logs)
        confianzas = [log.get("Confianza", 0.0) for log in exitosos_logs if isinstance(log.get("Confianza"), (int, float))]
        confianza_promedio = f"{(sum(confianzas) / len(confianzas)) * 100:.1f}%" if confianzas else "N/D"
    else:
        alertas_logs = 0
        confianza_promedio = "N/D"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Personas Registradas", f"{db_personas:,}")
    c2.metric("✍️ Muestras Base", f"{db_muestras:,}")
    c3.metric("🎯 Tasa de Éxito Global", tasa_exito_db)
    c4.metric("⚡ n8n Status", st.session_state.get("n8n_status", "🟢 Activo").split()[0])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🛡️ Alertas / Rechazos (Sesión)", f"{alertas_logs:,}")
    s2.metric("📈 Confianza Promedio (Aciertos)", confianza_promedio)
    s3.metric("🔍 Versión de Algoritmo", IDENTIFICATION_ALGORITHM_VERSION)
    s4.metric("📂 Eventos Registrados Hoy", f"{len(logs):,}")

    st.divider()
    st.markdown("### 📋 Historial de Auditoría en Tiempo Real")
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No hay eventos de auditoría registrados en la sesión actual.")

# --- PESTAÑA REGISTRAR FORMATO BASE ---
with tab_registrar:
    st.markdown("### 📝 Registro de Referencia Base")
    if "reg_file_key" not in st.session_state:
        st.session_state["reg_file_key"] = 0

    def reiniciar_registro():
        st.session_state["reg_nombre_input"] = ""
        st.session_state["reg_file_key"] += 1

    nombre_companero = st.text_input("Nombre del Usuario (Ej: Marta_Isabel)", key="reg_nombre_input")
    foto_base = st.file_uploader("Subir texto o palabra base registrada", type=["png", "jpg", "jpeg"], key=f"reg_file_{st.session_state['reg_file_key']}")

    img_base_pil = None
    evaluacion_base = None
    if foto_base is not None:
        try:
            img_base_pil = preparar_imagen_manuscrito(foto_base)
            evaluacion_base = evaluar_calidad_tecnica(img_base_pil)
            st.image(img_base_pil, caption="Vista previa de la referencia normalizada", width="stretch")
            mostrar_evaluacion_calidad(evaluacion_base, "Calidad de la referencia base")
        except ValueError as error:
            st.error(str(error), icon=":material/broken_image:")

    col_btn_reg1, col_btn_reg2 = st.columns(2)
    with col_btn_reg1:
        submitted = st.button("Guardar Referencia en DB", type="primary", use_container_width=True, disabled=evaluacion_base is not None and not evaluacion_base["apta"])
    with col_btn_reg2:
        if st.button("🔄 Reiniciar Registro", use_container_width=True, on_click=reiniciar_registro):
            st.rerun()

    if submitted:
        nombre_companero = " ".join(nombre_companero.split())
        if not nombre_companero or img_base_pil is None:
            st.warning("⚠️ Ingrese el nombre y cargue la imagen.")
        elif evaluacion_base is None or not evaluacion_base["apta"]:
            st.error("La referencia no cumple los criterios técnicos.")
        else:
            progreso_registro = st.progress(15, text="Preparando la referencia base...")
            with st.spinner("Generando embedding y registrando referencia base..."):
                try:
                    progreso_registro.progress(30, text="Validando escritura manuscrita con Gemini...")
                    es_valido, motivo = verificar_si_es_manuscrito(img_base_pil)
                    if not es_valido:
                        progreso_registro.progress(80, text="Registrando el rechazo en auditoría...")
                        registrar_en_n8n(nombre_companero, "ALERT_INVALID_REGISTRATION_IMAGE", 0.0, motivo)
                        progreso_registro.progress(100, text="Validación terminada: referencia rechazada.")
                        st.error(f"❌ Rechazado: La imagen no contiene un manuscrito válido. ({motivo})")
                    else:
                        progreso_registro.progress(50, text="Generando la huella grafomotora...")
                        vector_desc = generar_embedding_grafomotor(img_base_pil)
                        vector_str = "[" + ",".join(map(str, vector_desc)) + "]"

                        progreso_registro.progress(65, text="Guardando la imagen normalizada...")
                        nombre_seguro = re.sub(r"[^\w.-]+", "_", nombre_companero, flags=re.UNICODE).strip("._")[:80]
                        identificador = hashlib.sha256(nombre_companero.encode("utf-8")).hexdigest()[:10]
                        ruta_img = ASSETS_DIR / f"base_{nombre_seguro}_{identificador}.png"
                        img_base_pil.save(ruta_img, format="PNG", optimize=True)

                        progreso_registro.progress(80, text="Guardando la referencia en PostgreSQL...")
                        with obtener_conexion_db() as conn:
                            with conn.cursor() as cur:
                                cur.execute("INSERT INTO biometric_subjects (subject_id, full_name) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (nombre_companero, nombre_companero.replace("_", " ")))
                                cur.execute("""
                                    INSERT INTO biometric_samples (subject_id, feature_vector, embedding) 
                                    VALUES (%s, json_build_object('ruta_base', %s, 'fuente', 'multimodal')::jsonb, %s::vector);
                                """, (nombre_companero, str(ruta_img), vector_str))
                        cargar_metricas_db.clear()
                        progreso_registro.progress(100, text="Referencia registrada correctamente.")
                        st.success(f"✅ ¡Referencia base y embedding registrados con éxito para {nombre_companero}!")
                except Exception as e:
                    progreso_registro.empty()
                    st.error(f"No fue posible registrar la referencia: {e}")

# --- PESTAÑA VALIDACIÓN 1 A 1 ---
with tab_validar_1_1:
    st.markdown("### 🔐 Validación Biométrica 1 a 1 (Autenticación por Usuario)")
    st.markdown("<p style='color: #5b6f7c;'>Selecciona un usuario registrado e introduce su manuscrito para verificar exclusivamente su identidad.</p>", unsafe_allow_html=True)

    if "val_1_1_key" not in st.session_state:
        st.session_state["val_1_1_key"] = 0

    def reiniciar_validacion_1_1():
        st.session_state["val_1_1_key"] += 1

    try:
        referencias_1_1, _ = cargar_referencias_db()
        nombres_registrados = sorted(list(referencias_1_1.keys()))
    except Exception:
        nombres_registrados, referencias_1_1 = [], {}

    clave_validacion_1_1 = st.session_state["val_1_1_key"]
    usuario_seleccionado = st.selectbox(
        "Seleccionar Usuario Registrado",
        options=["-- Seleccione --"] + nombres_registrados,
        key=f"val_1_1_user_{clave_validacion_1_1}",
    )
    foto_1_1 = st.file_uploader(
        "Subir manuscrito del usuario para autenticar",
        type=["png", "jpg", "jpeg"],
        key=f"val_1_1_file_{clave_validacion_1_1}",
    )

    img_consulta_1_1 = None
    eval_1_1 = None
    if usuario_seleccionado != "-- Seleccione --" and foto_1_1 is not None:
        try:
            img_consulta_1_1 = preparar_imagen_manuscrito(foto_1_1)
            eval_1_1 = evaluar_calidad_tecnica(img_consulta_1_1)
            st.image(img_consulta_1_1, caption="Muestra a verificar", width="stretch")
            mostrar_evaluacion_calidad(eval_1_1, "Calidad técnica")
        except ValueError as err:
            st.error(str(err))

    col_btn_1_1_a, col_btn_1_1_b = st.columns(2)
    with col_btn_1_1_a:
        ejecutar_1_1 = st.button(
            "Verificar Identidad 1 a 1",
            type="primary",
            width="stretch",
            disabled=eval_1_1 is None or not eval_1_1["apta"],
        )
    with col_btn_1_1_b:
        if st.button("🔄 Reiniciar Validación", width="stretch", on_click=reiniciar_validacion_1_1):
            st.rerun()

    if ejecutar_1_1:
        progreso_1_1 = st.progress(15, text="Preparando la muestra de consulta...")
        with st.spinner(f"Verificando autenticidad para {usuario_seleccionado}..."):
            try:
                progreso_1_1.progress(35, text="Comparando a ciegas los rasgos grafomotores...")
                res_1_1 = verificar_autenticacion_1_a_1(img_consulta_1_1, usuario_seleccionado, referencias_1_1[usuario_seleccionado])
                progreso_1_1.progress(80, text="Generando el dictamen y la certificación...")

                matched = res_1_1["matched"]
                score = res_1_1["similarity_score"]
                conf = res_1_1["confidence"]
                similitud_vectorial = res_1_1["vector_similarity"]
                analisis = res_1_1["forensic_analysis"]

                estado = f"Autenticado: {usuario_seleccionado}" if matched else "Autenticación Fallida"
                estado_n8n = "AUTHENTICATED" if matched else "ALERT_UNAUTHORIZED"

                fecha_res = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                huella_cert = hashlib.sha256(f"{usuario_seleccionado}|{score}|{fecha_res}".encode()).hexdigest()[:16].upper()

                res_dict = {
                    "matched": matched,
                    "estado": estado,
                    "similarity_score": score,
                    "confidence": conf,
                    "forensic_analysis": analisis,
                    "best_match_subject": usuario_seleccionado if matched else "Desconocido",
                    "fecha": fecha_res,
                    "certificado_id": f"GID-1to1-{huella_cert}"
                }

                progreso_1_1.progress(90, text="Enviando evento de auditoría...")
                registrar_en_n8n(usuario_seleccionado, estado_n8n, conf, analisis)
                progreso_1_1.progress(100, text="Validación 1 a 1 terminada.")

                if matched:
                    st.success(f"✅ ¡Identidad autenticada correctamente para **{usuario_seleccionado}**!", icon="✅")
                else:
                    st.error("⚠️ La muestra no coincide con los patrones caligráficos del usuario registrado.", icon="⚠️")

                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("Decisión", "Autenticado" if matched else "Rechazado")
                c_b.metric("Similitud visual", f"{score * 100:.1f}%")
                c_c.metric("Confianza del análisis", f"{conf * 100:.1f}%")
                c_d.metric(
                    "Compatibilidad técnica (orientativa)",
                    f"{similitud_vectorial * 100:.1f}%",
                    help="Dato auxiliar de apariencia. No determina la autenticación 1 a 1.",
                )

                st.markdown(f"<div class='validation-result'><strong>Dictamen:</strong> {html.escape(analisis)}</div>", unsafe_allow_html=True)

                mostrar_diagnostico_comparacion(
                    img_consulta_1_1,
                    referencias_1_1[usuario_seleccionado],
                    usuario_seleccionado,
                )

                cert_html = generar_certificado_html(res_dict)
                st.download_button("Descargar Certificación 1 a 1 HTML", data=cert_html.encode("utf-8"), file_name=f"Cert_1to1_{res_dict['certificado_id']}.html", mime="text/html", type="primary", width="stretch")
            except Exception as error:
                progreso_1_1.empty()
                st.error(mensaje_error_validacion(error), icon=":material/cloud_off:")

# --- PESTAÑA IDENTIFICACIÓN GLOBAL (1 A N EN CASCADA) ---
with tab_validar_1_n:
    st.markdown("### 🔍 Identificación Global en Cascada (1 a N con Vectores)")
    st.markdown("<p style='color: #5b6f7c;'>Sube un manuscrito y la base de datos acotará los candidatos mediante pgvector antes del peritaje profundo de IA.</p>", unsafe_allow_html=True)

    if "val_file_key" not in st.session_state:
        st.session_state["val_file_key"] = 0

    def reiniciar_validacion():
        st.session_state["val_file_key"] += 1
        st.session_state.pop("resultado_validacion", None)

    foto_consulta = st.file_uploader("Subir manuscrito a identificar", type=["png", "jpg", "jpeg"], key=f"val_file_{st.session_state['val_file_key']}")

    img_consulta_pil = None
    evaluacion_consulta = None
    if foto_consulta is not None:
        try:
            img_consulta_pil = preparar_imagen_manuscrito(foto_consulta)
            evaluacion_consulta = evaluar_calidad_tecnica(img_consulta_pil)
            st.image(img_consulta_pil, caption="Vista previa del manuscrito", width="stretch")
            mostrar_evaluacion_calidad(evaluacion_consulta, "Calidad técnica de la consulta")
        except ValueError as error:
            st.error(str(error))

    col_btn_val1, col_btn_val2 = st.columns(2)
    with col_btn_val1:
        ejecutar_val = st.button("Ejecutar Identificación Vectorial (1 a N)", type="primary", use_container_width=True, disabled=evaluacion_consulta is not None and not evaluacion_consulta["apta"])
    with col_btn_val2:
        if st.button("🔄 Reiniciar", use_container_width=True, on_click=reiniciar_validacion):
            st.rerun()

    if ejecutar_val:
        if img_consulta_pil is not None and evaluacion_consulta and evaluacion_consulta["apta"]:
            progreso_validacion = st.progress(10, text="Preparando la muestra de consulta...")
            with st.spinner("Ejecutando filtrado vectorial y análisis pericial en cascada..."):
                try:
                    progreso_validacion.progress(30, text="Confirmando escritura manuscrita con Gemini...")
                    es_valido, motivo = verificar_si_es_manuscrito(img_consulta_pil)
                except Exception as error:
                    progreso_validacion.empty()
                    st.error(mensaje_error_validacion(error), icon=":material/cloud_off:")
                    es_valido = None

                if es_valido is False:
                    progreso_validacion.progress(100, text="Validación terminada: imagen rechazada.")
                    st.error(f"❌ Rechazado: La imagen no es un manuscrito válido. ({motivo})")
                elif es_valido is True:
                    progreso_validacion.progress(45, text="Cargando referencias desde PostgreSQL...")
                    refs, adv = cargar_referencias_db()
                    if not refs:
                        progreso_validacion.empty()
                        st.warning("⚠️ No hay referencias en la base de datos.")
                    else:
                        progreso_validacion.progress(60, text="Filtrando candidatos por similitud vectorial...")
                        try:
                            res_ia = identificar_manuscrito_en_bd_vectorial(img_consulta_pil, refs)
                        except Exception as error:
                            progreso_validacion.empty()
                            st.error(mensaje_error_validacion(error), icon=":material/cloud_off:")
                            st.stop()

                        if res_ia is None:
                            progreso_validacion.empty()
                            st.warning("⚠️ No se encontraron candidatos con embeddings disponibles.")
                            st.stop()

                        progreso_validacion.progress(85, text="Generando el dictamen pericial...")
                        matched = res_ia.get("matched", False)
                        sujeto = res_ia.get("best_match_subject", "Desconocido")
                        conf = res_ia.get("confidence", 0.0)
                        score = res_ia.get("similarity_score", 0.0)
                        similitud_vectorial = res_ia.get("vector_similarity", 0.0)
                        analisis = res_ia.get("forensic_analysis", "")

                        estado = f"Identificado: {sujeto}" if matched else "No reconocido"
                        estado_n8n = "AUTHENTICATED" if matched else "ALERT_UNAUTHORIZED"
                        fecha_res = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        huella_cert = hashlib.sha256(f"{sujeto}|{score}|{fecha_res}".encode()).hexdigest()[:16].upper()

                        st.session_state["resultado_validacion"] = {
                            "matched": matched,
                            "estado": estado,
                            "similarity_score": score,
                            "confidence": conf,
                            "vector_similarity": similitud_vectorial,
                            "forensic_analysis": analisis,
                            "best_match_subject": sujeto,
                            "fecha": fecha_res,
                            "certificado_id": f"GID-GLOB-{huella_cert}",
                            "reference_subject": sujeto if matched else (res_ia.get("rankings") or [["Desconocido"]])[0][0],
                            "rankings": res_ia.get("rankings", [])
                        }
                        progreso_validacion.progress(95, text="Enviando evento de auditoría...")
                        registrar_en_n8n(sujeto, estado_n8n, conf, analisis)
                        progreso_validacion.progress(100, text="Identificación terminada.")

    resultado_validacion = st.session_state.get("resultado_validacion")
    if resultado_validacion:
        if resultado_validacion["matched"]:
            st.success(f"✅ Identificado como: **{resultado_validacion['best_match_subject']}**")
        else:
            st.error("⚠️ El manuscrito no coincide con seguridad con ningún usuario registrado.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decisión", "Identificado" if resultado_validacion["matched"] else "No identificado")
        c2.metric("Similitud visual", f"{resultado_validacion['similarity_score'] * 100:.1f}%")
        c3.metric("Confianza del análisis", f"{resultado_validacion['confidence'] * 100:.1f}%")
        c4.metric(
            "Compatibilidad técnica (orientativa)",
            f"{resultado_validacion.get('vector_similarity', 0.0) * 100:.1f}%",
            help="Dato auxiliar de apariencia. No determina la identificación 1 a N.",
        )

        st.markdown(f"<div class='validation-result'><strong>Dictamen:</strong> {html.escape(resultado_validacion['forensic_analysis'])}</div>", unsafe_allow_html=True)

        ranking_resultado = resultado_validacion.get("rankings", [])[:CANTIDAD_CANDIDATOS_1_N]
        if ranking_resultado:
            st.markdown(f"#### Candidatos más próximos (Top {CANTIDAD_CANDIDATOS_1_N})")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Posición": posicion,
                        "Candidato": sujeto,
                        "Similitud": f"{puntaje * 100:.1f}%",
                    }
                    for posicion, (sujeto, puntaje) in enumerate(ranking_resultado, start=1)
                ]),
                hide_index=True,
                width="stretch",
            )

        sujeto_referencia = resultado_validacion.get("reference_subject")
        if img_consulta_pil is not None and sujeto_referencia:
            referencias_resultado, _ = cargar_referencias_db()
            referencia_resultado = referencias_resultado.get(sujeto_referencia)
            if referencia_resultado is not None:
                mostrar_diagnostico_comparacion(
                    img_consulta_pil,
                    referencia_resultado,
                    sujeto_referencia,
                    titulo=f"Referencia base más aproximada (1:N): {sujeto_referencia}",
                )
        
        cert_html = generar_certificado_html(resultado_validacion)
        st.download_button("Descargar Certificación 1 a N HTML", data=cert_html.encode("utf-8"), file_name=f"Cert_1toN_{resultado_validacion['certificado_id']}.html", mime="text/html", type="primary", use_container_width=True)