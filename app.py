import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# 🔥 FIX visual (quita barras blancas y espacios)
st.markdown("""
<style>
hr {display:none;}
[data-testid="stDivider"] {display:none;}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# BD
# =========================
USER = "postgres.nrtgdkhlyueerektkofu"
PASSWORD = "!Duquecito2021"
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

# =========================
# TEST CONEXIÓN
# =========================
try:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
except Exception as e:
    st.warning(f"BD: {e}")

# =========================
# MODELOS
# =========================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except:
        st.error("No se encontraron los modelos")
        return None, None, None

model, scaler, model_info = load_models()

# =========================
# UI
# =========================
st.title("🌸 Predictor de Especies de Iris")
st.caption("Predicción rápida y registro automático")

if model is not None:

    col1, col2 = st.columns(2)

    # =========================
    # INPUTS
    # =========================
    with col1:
        st.subheader("Datos")

        c1, c2 = st.columns(2)

        with c1:
            sepal_length = st.number_input("Sépalo L", 0.0, 10.0, 5.0, 0.1)
            sepal_width = st.number_input("Sépalo W", 0.0, 10.0, 3.0, 0.1)

        with c2:
            petal_length = st.number_input("Pétalo L", 0.0, 10.0, 4.0, 0.1)
            petal_width = st.number_input("Pétalo W", 0.0, 10.0, 1.0, 0.1)

        predict = st.button("Predecir")

    # =========================
    # RESULTADO
    # =========================
    with col2:
        st.subheader("Resultado")

        if predict:
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]

            target_names = model_info['target_names']
            predicted_species = target_names[prediction]
            confidence = float(max(probabilities))

            st.success(f"Especie: **{predicted_species}**")
            st.metric("Confianza", f"{confidence:.1%}")

            st.write("Probabilidades:")
            for species, prob in zip(target_names, probabilities):
                st.progress(float(prob), text=f"{species}: {prob:.1%}")

            # =========================
            # GUARDAR
            # =========================
            try:
                conn = get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                INSERT INTO ml.tb_iris 
                ("l_s", a_s, "l_p", a_p, prediccion, confidence)
                VALUES (%s, %s, %s, %s, %s, %s);
                """, (
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

                st.success("Guardado en BD")

            except Exception as e:
                st.error(f"Error: {e}")

# =========================
# HISTÓRICO
# =========================
st.subheader("Histórico")

try:
    conn = get_connection()

    df = pd.read_sql("""
    SELECT "l_s", a_s, "l_p", a_p, prediccion, confidence, created_at
    FROM ml.tb_iris
    ORDER BY created_at DESC;
    """, conn)

    conn.close()

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No hay registros")

except Exception as e:
    st.error(f"Error: {e}")
