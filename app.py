import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2

# =========================
# CONFIGURACIÓN DE LA APP
# =========================
st.set_page_config(
    page_title="Predictor de Iris",
    page_icon="🌸",
    layout="wide"
)

# =========================
# VARIABLES DE CONEXIÓN
# =========================
USER = "postgres.nrtgdkhlyueerektkofu"
PASSWORD = "!Duquecito2021"
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# =========================
# ESTILOS PERSONALIZADOS
# =========================
st.markdown("""
    <style>
        .main {
            background: linear-gradient(180deg, #fffafc 0%, #f8fbff 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .titulo-principal {
            font-size: 2.5rem;
            font-weight: 700;
            color: #7c3aed;
            margin-bottom: 0.3rem;
        }

        .subtitulo {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.2rem;
        }

        .resultado-box {
            background: linear-gradient(135deg, #ede9fe, #fdf2f8);
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid #ddd6fe;
            margin-top: 1rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(90deg, #8b5cf6, #ec4899);
            color: white;
            border: none;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }

        .stButton > button:hover {
            opacity: 0.92;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #ececec;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# FUNCIONES
# =========================
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

def get_database_time():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0]
    except Exception as e:
        return f"Error: {e}"

def save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO ml.tb_iris 
        ("l_s", a_s, "l_p", a_p, prediccion, confidence)
        VALUES (%s, %s, %s, %s, %s, %s);
        """

        cursor.execute(insert_query, (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species,
            confidence
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True, "✅ Guardado en la base de datos"
    except Exception as e:
        return False, f"Error al guardar: {e}"

def load_history():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        SELECT "l_s", a_s, "l_p", a_p, prediccion, confidence, created_at
        FROM ml.tb_iris
        ORDER BY created_at DESC;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if rows:
            df = pd.DataFrame(rows, columns=[
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width",
                "Predicción",
                "Confianza",
                "Fecha"
            ])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar histórico: {e}")
        return pd.DataFrame()

# =========================
# CARGA DE MODELOS
# =========================
model, scaler, model_info = load_models()

# =========================
# ENCABEZADO
# =========================
st.markdown('<div class="titulo-principal">🌸 Predictor de Especies de Iris</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">Ingresa las características de la flor para predecir su especie y visualizar el histórico de registros.</div>',
    unsafe_allow_html=True
)

db_time = get_database_time()
if isinstance(db_time, str) and db_time.startswith("Error"):
    st.warning("No se pudo obtener el estado de la base de datos.")
else:
    st.caption(f"🕒 Conexión activa con la base de datos | Fecha servidor: {db_time}")

# =========================
# CONTENIDO PRINCIPAL
# =========================
if model is not None:
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Datos de entrada")

        c1, c2 = st.columns(2)
        with c1:
            sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

        with c2:
            petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
            petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        predict_button = st.button("Predecir especie")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌼 Resultado")

        if predict_button:
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]

            target_names = model_info['target_names']
            predicted_species = target_names[prediction]
            confidence = float(max(probabilities))

            st.markdown('<div class="resultado-box">', unsafe_allow_html=True)
            st.success(f"Especie predicha: **{predicted_species}**")
            st.metric("Confianza", f"{confidence:.1%}")
            st.markdown("**Probabilidades por especie:**")

            for species, prob in zip(target_names, probabilities):
                st.progress(float(prob), text=f"{species}: {prob:.1%}")

            st.markdown('</div>', unsafe_allow_html=True)

            saved, message = save_prediction(
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
                predicted_species,
                confidence
            )

            if saved:
                st.success(message)
            else:
                st.error(message)
        else:
            st.info("Completa los campos y presiona el botón para ver la predicción.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HISTÓRICO
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Histórico de Predicciones")

df_history = load_history()

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("No hay registros aún.")

st.markdown('</div>', unsafe_allow_html=True)
