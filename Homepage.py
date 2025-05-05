import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import geopandas as gpd
from shapely.geometry import Point
import re
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="./images/icon.png",
    layout="wide"
)

query_params = st.query_params
page = query_params.get("page", "Home")
st.session_state["current_page"] = page

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model_path = 'telwave-customer-churn-prediction3.sav'
model = load_model(model_path)


with st.sidebar:

    st.sidebar.image("./images/telwave-technology.png", use_column_width=True)

   # Custom CSS for button-style nav links
    st.markdown("""
        <style>
            .nav-button {
                display: block;
                padding: 10px 16px;
                margin: 8px 0;
                background-color: rgba(0,0,0,.05);
                color: #fff !important;
                text-decoration: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 500;
                transition: background-color 0.2s ease;
            }
            .nav-button:hover {
                background-color: rgba(0,0,0,.1);
                text-decoration:none;
            }
            .nav-button.active {
                background-color: #fbcd2b;
                color: #212121 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    

    def nav_button(label):
        is_active = (label == page)
        btn_class = "nav-button active" if is_active else "nav-button"
        html = f'''
            <a href="/?page={label}" class="{btn_class}" target="_self">{label}</a>
        '''
        st.markdown(html.strip(), unsafe_allow_html=True)

    nav_button("Home")
    nav_button("Bulk Prediction")
    nav_button("Single Prediction")

def add_county(df):
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['Longitude'], df['Latitude'])],
        crs="EPSG:4326"
    )
    gdf_counties = gpd.read_file('California_Counties.geojson').to_crs("EPSG:4326")
    gdf_joined = gpd.sjoin(
        gdf_pts,
        gdf_counties[['NAME', 'geometry']],
        how='left',
        predicate='within'
    )
    df['County'] = gdf_joined['NAME'].values
    return df

def process_initial(df):
    df = df.copy()
    df = add_county(df)

    drop_cols = ['CustomerID','Latitude','Longitude','ZipCode','City',
                 'ChurnCategory','ChurnReason','ChurnScore','Country','ReferredFriend']
    df_model = df.drop(columns=[col for col in drop_cols if col in df.columns])

    model_features = model.feature_names_in_

    for col in model_features:
        if col not in df_model.columns:
            df_model[col] = 0
    print(df_model.dtypes)    
    numeric_cols = ['AvgMonthlyLongDistanceCharges', 'CLTV', 'MonthlyCharges', 
                'AvgMonthlyGBDownload', 'Tenure', 'Age']

    for col in numeric_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
    if 'SeniorCitizen' in df_model.columns:
        df_model['SeniorCitizen'] = df_model['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    if 'SatisfactionScore' in df_model.columns:
        df_model['SatisfactionScore'] = pd.to_numeric(df_model['SatisfactionScore'], errors='coerce').fillna(1).astype(int)
        df_model['SatisfactionScore'] = df_model['SatisfactionScore'].apply(lambda x: x if x in [1, 2, 3, 4, 5] else df_model['SatisfactionScore'].median())
    if 'AvgMonthlyLongDistanceCharges' in df_model.columns:
        df_model['AvgMonthlyLongDistanceCharges'] = pd.to_numeric(df_model['AvgMonthlyLongDistanceCharges'], errors='coerce')
    def convert_referrals(val):
        if val == 'Yes':
            return 1
        elif val == 'No':
            return 0
        else:
            return val  # asumsikan val udah numerik

    df_model['Referrals'] = df_model['Referrals'].apply(convert_referrals).astype(float)
    
    df_model = df_model[model_features]
    

    df['churn_prob'] = model.predict_proba(df_model)[:, 1]
    return df

# ===== FUNGSI: Update churn & priority dari threshold =====
def apply_threshold(df, threshold):
    df = df.copy()
    df['churn'] = (df['churn_prob'] > threshold).astype(int)
    df['priority'] = df['churn_prob'] > 0.8
    return df

def create_gauge_chart(probability, threshold=0.31):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Churn Probability (%)"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "green"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ],
        },
    ))


    st.plotly_chart(fig)

# Display pages
if page == "Home":
    # Create layout
    with st.container():
        col1, col2 = st.columns([1, 2])
    
        with col1:
            st.image(
                "./images/lg.png",
            )
    
        with col2:
            st.markdown(
                """
                <div style="display: flex; flex-direction: column; justify-content: center; height: 32vh;margin-top:50px">
                    <h1 style="font-size: 60px;color:#fbcd2b;margin-left:20px">Welcome to Telwave Technology Churn Prediction Tool</h1>
                    <p style="font-size: 20px;color:#fff;margin-left:20px;text-align:justify">Leverage advanced analytics to anticipate     customer churn before it happens. Our tool helps you identify at-risk customers, understand behavioral patterns, and     take proactive steps to improve retention and drive long-term growth.</p>
                </div>
                """,
                unsafe_allow_html=True
        )
    
    
    st.write("")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 To use this tool, go to the left and select a page.")
    
    # Three Key Sections Below
    col3, col4, col5 = st.columns([2, 2, 2])
    
    with col3:
        st.markdown("#### 🔍 Customer Churn Analysis")
        st.markdown("---")
        st.write("Identify customer behavior patterns and predict the likelihood of churn within the next month.")
    
    with col4:
        st.markdown("#### 🤖 Powered by Logistic Regression")
        st.markdown("---")
        st.write("Using Logistic Regression, a powerful machine learning model, to accurately predict customer churn based on historical data.")
    
    with col5:
        st.markdown("#### 📈 Strategic Insights")
        st.markdown("---")
        st.write("Enable proactive decision-making by understanding customer behavior and minimizing the risk of churn within the next     month.")
    
    # Disclaimer and Additional Information
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #7F8C8D;">
        <p>Disclaimer: This tool provides insights based on historical data and may require updates for accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('''
    For details on how the model works, visit: [Click Here]()
    
    Created by: Team Alpha
''')
elif page == "Bulk Prediction":
    st.title("📊 Customer Prediction")    


    uploaded_file = st.file_uploader("Upload Data to Predict (CSV)", type=['csv'])
    
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        with st.spinner("🔍 Memproses data awal..."):
            st.session_state.df_original = process_initial(df_uploaded)
        st.success("✅ Data berhasil diproses!")
    
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    
    if st.session_state.df_original is not None:
        threshold = st.slider("🎯 Atur Threshold Churn (default 0.31)", 0.0, 1.0, 0.31, step=0.01)
        df_result = apply_threshold(st.session_state.df_original, threshold)

        df_churn = df_result[df_result['churn'] == 1]
        df_priority = df_churn[df_churn['priority']]

         # =======================
        # 🔢 Visualisasi Overview
        # =======================
        churn_counts = df_result['churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        churn_counts['Churn'] = churn_counts['Churn'].map({1: 'Yes', 0: 'No'})

        col_bar, col_pie = st.columns(2)

        with col_bar:
            st.markdown("### 📊 Distribusi Churn")
            fig_bar = px.bar(churn_counts, x='Churn', y='Count', color='Churn',
                            color_discrete_map={'Yes': 'red', 'No': 'green'},
                            text='Count', title='Jumlah Pelanggan Churn vs Tidak')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            st.markdown("### 🥧 Proporsi Churn")
            fig_pie = px.pie(churn_counts, values='Count', names='Churn',
                            color='Churn',
                            color_discrete_map={'Yes': 'red', 'No': 'green'},
                            title='Proporsi Pelanggan Churn')
            st.plotly_chart(fig_pie, use_container_width=True)

        # =======================
        # 📊 Data Tables 3 Kolom
        # =======================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🧾 Semua Data")
            st.dataframe(df_result, use_container_width=True)
            st.download_button("⬇️ Download Semua", df_result.to_csv(index=False), "all_data.csv", use_container_width=True)

        with col2:
            st.markdown("### ⚠️ Pelanggan Churn")
            st.dataframe(df_churn, use_container_width=True)
            st.download_button("⬇️ Download Churn", df_churn.to_csv(index=False), "churn_data.csv", use_container_width=True)

        with col3:
            st.markdown("### 🔥 Prioritas Tinggi (Prob > 0.8)")
            st.dataframe(df_priority, use_container_width=True)
            st.download_button("⬇️ Download Prioritas", df_priority.to_csv(index=False), "priority_data.csv", use_container_width=True)
elif page == 'Single Prediction':
    def plot_odometer(probability):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.barh(0, probability, color='green', height=0.3)
        ax.text(probability / 2, 0.5, f'{probability*100:.2f}%', ha='center', va='center', fontsize=15, color='white')
        
        st.pyplot(fig)

    def predict_churn(input_data):
               
        churn_prob = model.predict_proba(input_data)[:, 1]  # Probabilitas churn (class 1)
        return churn_prob[0]

    # Streamlit UI untuk input data
    st.title("Customer Churn Prediction")

    with st.form(key="input_form"):
        # Membuat 3 kolom untuk input
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 19, 80)
            population = st.number_input("Population", min_value=11, max_value=105285)
            referrals = st.number_input("Referrals", min_value=0, max_value=11)
            satisfaction_score = st.slider("Satisfaction Score", 1, 5)

        with col2:
            tenure = st.slider("Tenure (Months)", 1, 72)
            monthly_charges = st.number_input("Monthly Charges (USD)", min_value=18.40, max_value=118.65)
            avg_monthly_gb_download = st.number_input("Avg Monthly GB Download", min_value=0, max_value=85)
            avg_monthly_long_distance_charges = st.number_input("Avg Monthly Long Distance Charges (USD)", min_value=0.00, max_value=49.99)

        with col3:
            cltv = st.number_input("Customer Lifetime Value (USD)", min_value=2003, max_value=6500)
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
            married = st.selectbox("Married", ["Yes", "No"])

        with col1:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])

        with col3:
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Credit card (automatic)", "Electronic check", "Bank transfer (automatic)", "Mailed check"])

        with col1:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col2:
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            referred_friend = st.selectbox("Referred Friend", ["Yes", "No"])
            county = st.selectbox("County", ["Los Angeles County", "San Diego County", "Orange County", "San Francisco County"])

        # Submit button untuk form
        submit_button = st.form_submit_button("Predict Churn")

        if submit_button:
            # Mempersiapkan input data untuk prediksi
            input_data = pd.DataFrame({
                'Age': [age],
                'Population': [population],
                'Referrals': [referrals],
                'SatisfactionScore': [satisfaction_score],
                'Tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'AvgMonthlyGBDownload': [avg_monthly_gb_download],
                'AvgMonthlyLongDistanceCharges': [avg_monthly_long_distance_charges],
                'CLTV': [cltv],
                'Gender': [gender],
                'SeniorCitizen': [senior_citizen],
                'Married': [married],
                'Partner': [partner],
                'Dependents': [dependents],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'StreamingMusic': [streaming_music],
                'UnlimitedData': [unlimited_data],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'ReferredFriend': [referred_friend],
                'County': [county]
            })

            # Prediksi probabilitas churn
            churn_prob = predict_churn(input_data)

            # Menampilkan probabilitas churn
            st.subheader(f"Predicted Churn Probability: {churn_prob * 100:.2f}%")

            # Visualisasi dengan gauge chart (odometer) dengan threshold line
            create_gauge_chart(churn_prob)
            

