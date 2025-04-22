import streamlit as st

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="./images/icon.png",
    layout="wide"
)

query_params = st.query_params
page = query_params.get("page", "Home")
st.session_state["current_page"] = page

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
    nav_button("Customer Prediction")



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
        st.markdown("#### 🤖 Powered by XGBoost")
        st.markdown("---")
        st.write("Using XGBoost, a powerful machine learning model, to accurately predict customer churn based on historical data.")
    
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
elif page == "Customer Prediction":
    st.title("Customer Prediction")

