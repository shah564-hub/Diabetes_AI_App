import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Diabetes AI Predictor", page_icon="🩺", layout="wide")


# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""


# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"])


# -------------------------------
# PREMIUM CSS
# -------------------------------
if theme == "🌙 Dark Mode":
    bg = "radial-gradient(circle at top left, #1b2a4a 0%, #0b1220 45%, #070b13 100%)"
    card_bg = "rgba(255, 255, 255, 0.06)"
    border = "1px solid rgba(255,255,255,0.10)"
    text_color = "#eaf1ff"
    sub_text = "rgba(234,241,255,0.75)"
    shadow = "0px 12px 30px rgba(0,0,0,0.45)"
else:
    bg = "linear-gradient(135deg, #fff1eb 0%, #ace0f9 100%)"
    card_bg = "rgba(255,255,255,0.88)"
    border = "1px solid rgba(0,0,0,0.06)"
    text_color = "#0f172a"
    sub_text = "rgba(15,23,42,0.70)"
    shadow = "0px 10px 22px rgba(15,23,42,0.12)"

st.markdown(
    f"""
<style>

/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
}}

.stApp {{
    background: {bg};
    color: {text_color};
}}

/* Remove top padding */
.block-container {{
    padding-top: 1.2rem !important;
}}

/* Titles */
.main-title {{
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: {text_color};
    letter-spacing: 0.5px;
}}

.sub-title {{
    text-align: center;
    font-size: 15px;
    color: {sub_text};
    margin-top: -12px;
}}

/* Glass cards */
.card {{
    background: {card_bg};
    border: {border};
    padding: 22px;
    border-radius: 22px;
    box-shadow: {shadow};
    margin: 10px 0px;
    backdrop-filter: blur(10px);
}}

/* Buttons */
div.stButton > button {{
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 16px;
    font-weight: 700;
    padding: 12px 18px;
    border-radius: 14px;
    border: none;
    transition: 0.25s;
    width: 100%;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0px 10px 18px rgba(0,0,0,0.25);
}}

/* Tabs readability */
.stTabs [data-baseweb="tab-list"] {{
    gap: 10px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: 600;
}}

/* Slider label visibility */
label {{
    color: {text_color} !important;
    font-weight: 600 !important;
}}

/* Table readability */
thead tr th {{
    color: {text_color} !important;
}}
tbody tr td {{
    color: {text_color} !important;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(12px);
}}
</style>
""",
    unsafe_allow_html=True
)


# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    return pd.read_csv(url, names=columns)


data = load_data()


# -------------------------------
# TRAIN MODEL
# -------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)


# -------------------------------
# HEALTH TIPS
# -------------------------------
def get_health_tips(pred):
    if pred == 1:
        return [
            "Reduce sugary foods & soft drinks",
            "Walk 30 minutes daily",
            "Maintain healthy BMI (exercise + diet)",
            "Regularly check blood sugar level",
            "Consult a doctor for guidance"
        ]
    return [
        "Maintain a balanced diet",
        "Stay physically active",
        "Maintain healthy body weight",
        "Drink enough water daily",
        "Do regular health checkups"
    ]


# -------------------------------
# PDF CLEAN TEXT
# -------------------------------
def clean_for_pdf(text: str) -> str:
    # remove emojis/unicode for safe PDF output
    remove_list = ["✅", "⚠️", "🎉", "🩺", "📌", "📊", "💡", "👋", "🌙", "🌞", "🔍", "📄"]
    for ch in remove_list:
        text = text.replace(ch, "")
    return text.strip()


# -------------------------------
# PDF GENERATION
# -------------------------------
def generate_pdf_report(user_name, record):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(200, 10, clean_for_pdf("Diabetes Prediction Report"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 12)
    pdf.ln(10)
    pdf.cell(200, 8, clean_for_pdf(f"Name: {user_name}"), ln=True)
    pdf.cell(200, 8, clean_for_pdf(f"Date & Time: {record['Time']}"), ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(200, 8, clean_for_pdf("Patient Inputs:"), ln=True)

    pdf.set_font("Helvetica", "", 12)
    for key in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]:
        pdf.cell(200, 8, clean_for_pdf(f"{key}: {record[key]}"), ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(200, 8, clean_for_pdf("Prediction:"), ln=True)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(200, 8, clean_for_pdf(f"Result: {record['Prediction']}"), ln=True)
    pdf.cell(200, 8, clean_for_pdf(f"Probability: {record['Probability(%)']}%"), ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(200, 8, clean_for_pdf("Health Tips:"), ln=True)

    tips = get_health_tips(1 if record["Prediction"] == "Diabetes" else 0)
    pdf.set_font("Helvetica", "", 12)
    for tip in tips:
        pdf.cell(200, 8, clean_for_pdf(f"- {tip}"), ln=True)

    file_name = "diabetes_report.pdf"
    pdf.output(file_name)
    return file_name


# -------------------------------
# LOGIN UI (DEMO)
# -------------------------------
def login_register_ui():
    st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Premium UI • Dark Mode • PDF Report • Streamlit App</p>", unsafe_allow_html=True)
    st.write(" ")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔐 Login / Register (Demo)")

    t1, t2 = st.tabs(["Login", "Register"])

    with t1:
        username = st.text_input("Username", placeholder="Enter your name")
        _ = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login"):
            if username.strip() == "":
                st.warning("Please enter username")
            else:
                st.session_state.logged_in = True
                st.session_state.user_name = username
                st.success(f"Welcome, {username} ✅")
                st.rerun()

    with t2:
        new_user = st.text_input("Create Username", placeholder="Choose a username")
        _ = st.text_input("Create Password", type="password", placeholder="Choose password")
        if st.button("Register"):
            if new_user.strip() == "":
                st.warning("Please enter username")
            else:
                st.success("Registered Successfully ✅ (Demo only)")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# STOP IF NOT LOGGED IN
# -------------------------------
if not st.session_state.logged_in:
    login_register_ui()
    st.stop()


# -------------------------------
# MAIN UI
# -------------------------------
st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction AI</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p class='sub-title'>Welcome, <b>{st.session_state.user_name}</b> 👋 • Theme: <b>{theme}</b> • PDF Report Enabled</p>",
    unsafe_allow_html=True
)

st.sidebar.success(f"✅ Logged in as: {st.session_state.user_name}")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user_name = ""
    st.session_state.history = []
    st.rerun()


left, right = st.columns([1, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Enter Patient Details")

    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 300, 120)
    bp = st.slider("Blood Pressure", 0, 200, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 28.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 1, 120, 25)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    if st.button("🔍 Predict Now"):
        with st.spinner("Running model prediction..."):
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

        st.write("Probability of Diabetes Risk")
        st.progress(int(prob * 100))

        if pred == 1:
            result_text = "Diabetes"
            st.error(f"⚠️ High chance of Diabetes ({prob*100:.2f}%)")
        else:
            result_text = "No Diabetes"
            st.success(f"✅ Low chance of Diabetes ({prob*100:.2f}%)")

        st.info(f"Model Accuracy: **{accuracy*100:.2f}%**")

        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Age": age,
            "Prediction": result_text,
            "Probability(%)": round(prob * 100, 2)
        }
        st.session_state.history.append(record)

        st.subheader("💡 Health Tips")
        tips = get_health_tips(pred)
        for t in tips:
            st.write("• " + t)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# TABS
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🧾 History", "📌 Dataset", "📈 Charts", "ℹ️ About"])

with tab1:
    if len(st.session_state.history) == 0:
        st.warning("No predictions yet. Click Predict Now ✅")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Report", csv, "history_report.csv", "text/csv")

        latest = st.session_state.history[-1]
        if st.button("📄 Generate PDF for Latest Prediction"):
            pdf_file = generate_pdf_report(st.session_state.user_name, latest)
            with open(pdf_file, "rb") as f:
                st.download_button("⬇️ Download PDF Report", f, "diabetes_report.pdf", "application/pdf")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.success("History cleared ✅")
            st.rerun()

with tab2:
    st.dataframe(data.head(20), use_container_width=True)

with tab3:
    st.subheader("Outcome Distribution")
    st.bar_chart(data["Outcome"].value_counts())

with tab4:
    st.markdown(f"""
    ✅ **Project:** Diabetes Prediction AI Web App  
    ✅ **User:** {st.session_state.user_name}  
    ✅ **Tech:** Streamlit, Python, Pandas, NumPy, Scikit-learn, fpdf2  
    ✅ **Model:** Random Forest Classifier  
    ✅ **Features:** Premium UI, Dark Mode, History, CSV, PDF Report, Health Tips  
    """)

st.markdown("</div>", unsafe_allow_html=True)

st.caption("✨ Built by Shahida | AI Internship Project | Streamlit App")
