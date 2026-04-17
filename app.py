import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Predictor de Iris",
    page_icon="🌸",
    layout="wide"
)

USER = "postgres.nrtgdkhlyueerektkofu"
PASSWORD = "!Duquecito2021"
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# =========================
# ESTILOS (COMPACTOS)
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #fffafc 0%, #f8fbff 100%);
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.titulo-principal {
    font-size: 2.2rem;
    font-weight: 700;
    color: #7c3aed;
    margin-bottom: 0.2rem;
}

.subtitulo {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 1rem;
}

.card {
    background-color: white;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 0.6rem;
}

.resultado-box {
    background: linear-gradient(135deg, #ede9fe, #fdf2f8);
    padding: 0.8rem;
    border-radius: 12px;
    border: 1px solid #ddd6fe;
    margin-top: 0.5rem;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    color: white;
    border: none;
    font-weight: 600;
    padding: 0.5rem;
}

.stButton > button:hover {
    opacity: 0.9;
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
    except:
        st.error("Error cargando modelo")
        return None, None, None

def save_prediction(data):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO ml.tb_iris 
        ("l_s", a_s, "l_p", a_p, prediccion, confidence)
        VALUES (%s, %s, %s, %s, %s, %s);
        """, data)

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(e)
        return False

def load_history():
    try:
        conn = get_connection()
        df = pd.read_sql("""
        SELECT "l_s", a_s, "l_p", a_p, prediccion, confidence, created_at
        FROM ml.tb_iris
        ORDER BY created_at DESC;
        """, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# =========================
# APP
# =========================
model, scaler, model_info = load_models()

st.markdown('<div class="titulo-principal">🌸 Predictor de Iris</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Predicción rápida y registro automático</div>', unsafe_allow_html=True)

if model:
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Datos")

        c1, c2 = st.columns(2)
        with c1:
            sl = st.number_input("Sépalo L", 0.0, 10.0, 5.0, 0.1)
            sw = st.number_input("Sépalo W", 0.0, 10.0, 3.0, 0.1)
        with c2:
            pl = st.number_input("Pétalo L", 0.0, 10.0, 4.0, 0.1)
            pw = st.number_input("Pétalo W", 0.0, 10.0, 1.0, 0.1)

        predict = st.button("Predecir")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Resultado")

        if predict:
            X = np.array([[sl, sw, pl, pw]])
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]

            names = model_info['target_names']
            specie = names[pred]
            conf = float(max(probs))

            st.markdown('<div class="resultado-box">', unsafe_allow_html=True)
            st.success(f"{specie}")
            st.metric("Confianza", f"{conf:.1%}")

            for n, p in zip(names, probs):
                st.progress(float(p))

            st.markdown('</div>', unsafe_allow_html=True)

            save_prediction((sl, sw, pl, pw, specie, conf))

        st.markdown('</div>', unsafe_allow_html=True)

# HISTÓRICO (SIN ESPACIOS EXTRA)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Histórico")

df = load_history()
if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.info("Sin datos")

st.markdown('</div>', unsafe_allow_html=True)
