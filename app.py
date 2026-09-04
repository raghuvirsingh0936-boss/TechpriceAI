import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TechPrice AI",
    page_icon="💻",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #0A3D62;
    font-size: 42px;
    font-weight: 800;
}

h2, h3 {
    color: #17324D;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e5ea;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 17px;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        135deg,
        #232526,
        #414345
    );
}

section[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

rf_model = joblib.load("techprice_rf_model.pkl")
preprocessor = joblib.load("techprice_preprocessor.pkl")

df = pd.read_csv("Laptop.csv")


# =========================================================
# DEAL EVALUATION
# =========================================================

def evaluate_laptop_deal(actual_price, predicted_price):

    difference = actual_price - predicted_price

    if difference <= -100:
        status = "🔥 Great Deal"

    elif difference <= 100:
        status = "✅ Fair Price"

    elif difference <= 300:
        status = "⚠️ Slightly Expensive"

    else:
        status = "❌ Expensive"

    return status, difference


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💻 TechPrice AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 Price Prediction",
        "📊 Price Analysis",
        "🧠 Model Performance",
        "ℹ️ About"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("💻 TechPrice AI")

    st.markdown(
        "### 🤖 AI-Based Laptop Price Prediction & Deal Evaluation"
    )

    st.write(
        "TechPrice AI uses Machine Learning to predict "
        "laptop prices based on hardware specifications."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "🤖 **AI Prediction**\n\n"
            "Predict the expected price of a laptop."
        )

    with col2:
        st.success(
            "💰 **Deal Evaluation**\n\n"
            "Check whether the selling price is a good deal."
        )

    with col3:
        st.warning(
            "📊 **Price Analysis**\n\n"
            "Explore laptop pricing patterns."
        )

    st.divider()

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💻 Total Laptops",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "🏷️ Brands",
            df["Company"].nunique()
        )

    with col3:
        st.metric(
            "💰 Average Price",
            f"€ {df['Price_euros'].mean():,.0f}"
        )

    with col4:
        st.metric(
            "📈 Maximum Price",
            f"€ {df['Price_euros'].max():,.0f}"
        )

    st.divider()

    st.subheader("🚀 How It Works")

    st.write("""
    **1️⃣ Enter Laptop Specifications**

    Enter the brand, laptop type, processor, GPU, RAM,
    storage, display resolution and weight.

    **2️⃣ AI Predicts the Price**

    The Random Forest Machine Learning model estimates
    the expected laptop price.

    **3️⃣ Compare the Price**

    Enter the current selling price.

    **4️⃣ Get Deal Evaluation**

    The system tells whether the laptop is a
    Great Deal, Fair Price, Slightly Expensive, or Expensive.
    """)


# =========================================================
# PRICE PREDICTION
# =========================================================

elif page == "🤖 Price Prediction":

    st.title("🤖 Laptop Price Prediction")

    st.write(
        "Enter laptop specifications to predict its expected price."
    )

    st.divider()

    # Laptop information

    st.subheader("💻 Laptop Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        company = st.selectbox(
            "🏷️ Brand",
            sorted(df["Company"].unique())
        )

    with col2:
        typename = st.selectbox(
            "💻 Laptop Type",
            [
                "Notebook",
                "Ultrabook",
                "Netbook",
                "Gaming",
                "2 in 1 Convertible",
                "Workstation"
            ]
        )

    with col3:
        inches = st.number_input(
            "📏 Screen Size (Inches)",
            min_value=10.0,
            max_value=20.0,
            value=15.6,
            step=0.1
        )

    # Hardware

    st.subheader("⚙️ Hardware")

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu_brand = st.selectbox(
            "🧠 CPU Brand",
            [
                "Intel",
                "AMD",
                "Samsung",
                "ARM"
            ]
        )

    with col2:
        gpu_brand = st.selectbox(
            "🎮 GPU Brand",
            [
                "Intel",
                "Nvidia",
                "AMD",
                "ARM"
            ]
        )

    with col3:
        opsys = st.selectbox(
            "🪟 Operating System",
            [
                "Windows 10",
                "Windows 10 S"
            ]
        )

    # RAM and storage

    col1, col2, col3 = st.columns(3)

    with col1:
        ram = st.selectbox(
            "🧮 RAM (GB)",
            [2, 4, 6, 8, 12, 16, 24, 32, 64]
        )

    with col2:
        ssd = st.number_input(
            "💾 SSD Storage (GB)",
            min_value=0,
            max_value=4096,
            value=256,
            step=128
        )

    with col3:
        hdd = st.number_input(
            "💽 HDD Storage (GB)",
            min_value=0,
            max_value=4096,
            value=0,
            step=128
        )

    # Display

    st.subheader("🖥️ Display")

    col1, col2 = st.columns(2)

    with col1:
        resolution_width = st.number_input(
            "Resolution Width",
            min_value=800,
            max_value=5000,
            value=1920,
            step=10
        )

    with col2:
        resolution_height = st.number_input(
            "Resolution Height",
            min_value=600,
            max_value=4000,
            value=1080,
            step=10
        )

    # Weight

    weight = st.number_input(
        "⚖️ Weight (Kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.8,
        step=0.1
    )

    # Selling price

    st.subheader("💰 Current Selling Price")

    actual_price = st.number_input(
        "Selling Price (€)",
        min_value=100,
        max_value=10000,
        value=700,
        step=50
    )

    # Calculated features

    total_storage = ssd + hdd

    pixels = resolution_width * resolution_height

    st.divider()

    # Prediction button

    if st.button(
        "🚀 Predict Laptop Price",
        use_container_width=True
    ):

        new_laptop = pd.DataFrame([{

            "Company": company,
            "TypeName": typename,
            "Inches": inches,
            "CPU_Brand": cpu_brand,
            "GPU_Brand": gpu_brand,
            "OpSys": opsys,
            "Ram (GB)": ram,
            "SSD_GB": ssd,
            "HDD_GB": hdd,
            "Total_Storage_GB": total_storage,
            "Resolution_Width": resolution_width,
            "Resolution_Height": resolution_height,
            "Pixels": pixels,
            "Weight (Kg)": weight

        }])

        new_laptop_encoded = preprocessor.transform(
            new_laptop
        )

        prediction = rf_model.predict(
            new_laptop_encoded
        )[0]

        status, difference = evaluate_laptop_deal(
            actual_price,
            prediction
        )

        st.divider()

        st.subheader("🎯 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🤖 AI Estimated Price",
                f"€ {prediction:,.2f}"
            )

        with col2:
            st.metric(
                "💰 Selling Price",
                f"€ {actual_price:,.2f}"
            )

        with col3:
            st.metric(
                "📊 Price Difference",
                f"€ {difference:,.2f}"
            )

        st.divider()

        if status == "🔥 Great Deal":

            st.success(
                f"### {status}\n\n"
                "This laptop is significantly cheaper "
                "than the AI estimated price."
            )

        elif status == "✅ Fair Price":

            st.info(
                f"### {status}\n\n"
                "The selling price is close to the "
                "AI estimated price."
            )

        elif status == "⚠️ Slightly Expensive":

            st.warning(
                f"### {status}\n\n"
                "The laptop is somewhat more expensive "
                "than the AI estimated price."
            )

        else:

            st.error(
                f"### {status}\n\n"
                "The selling price is considerably higher "
                "than the AI estimated price."
            )


# =========================================================
# PRICE ANALYSIS
# =========================================================

elif page == "📊 Price Analysis":

    st.title("📊 Laptop Price Analysis")

    st.write(
        "Explore laptop pricing patterns in the dataset."
    )

    st.divider()

    # KPIs

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💰 Average Price",
            f"€ {df['Price_euros'].mean():,.2f}"
        )

    with col2:
        st.metric(
            "📈 Maximum Price",
            f"€ {df['Price_euros'].max():,.2f}"
        )

    with col3:
        st.metric(
            "📉 Minimum Price",
            f"€ {df['Price_euros'].min():,.2f}"
        )

    st.divider()

    # Filter

    st.subheader("🔍 Filter by Brand")

    selected_brands = st.multiselect(
        "Select Laptop Brands",
        sorted(df["Company"].unique()),
        default=sorted(df["Company"].unique())
    )

    if selected_brands:

        filtered_df = df[
            df["Company"].isin(selected_brands)
        ]

        # Brand analysis

        brand_price = (
            filtered_df.groupby("Company")["Price_euros"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_brand = px.bar(
            brand_price,
            x="Company",
            y="Price_euros",
            title="Average Laptop Price by Brand",
            labels={
                "Company": "Brand",
                "Price_euros": "Average Price (€)"
            }
        )

        fig_brand.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig_brand,
            use_container_width=True
        )

        # RAM analysis

        ram_price = (
            filtered_df.groupby("Ram (GB)")["Price_euros"]
            .mean()
            .sort_index()
            .reset_index()
        )

        fig_ram = px.line(
            ram_price,
            x="Ram (GB)",
            y="Price_euros",
            markers=True,
            title="Average Laptop Price by RAM",
            labels={
                "Ram (GB)": "RAM (GB)",
                "Price_euros": "Average Price (€)"
            }
        )

        st.plotly_chart(
            fig_ram,
            use_container_width=True
        )

        # Type analysis

        type_price = (
            filtered_df.groupby("TypeName")["Price_euros"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_type = px.bar(
            type_price,
            x="TypeName",
            y="Price_euros",
            title="Average Laptop Price by Type",
            labels={
                "TypeName": "Laptop Type",
                "Price_euros": "Average Price (€)"
            },
            text_auto=".0f"
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True
        )

        # CPU analysis

        cpu_df = filtered_df.copy()

        cpu_df["CPU_Brand"] = (
            cpu_df["Cpu"].str.split().str[0]
        )

        cpu_price = (
            cpu_df.groupby("CPU_Brand")["Price_euros"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_cpu = px.bar(
            cpu_price,
            x="CPU_Brand",
            y="Price_euros",
            title="Average Laptop Price by CPU Brand",
            labels={
                "CPU_Brand": "CPU Brand",
                "Price_euros": "Average Price (€)"
            },
            text_auto=".0f"
        )

        st.plotly_chart(
            fig_cpu,
            use_container_width=True
        )

        # GPU analysis

        gpu_df = filtered_df.copy()

        gpu_df["GPU_Brand"] = (
            gpu_df["Gpu"].str.split().str[0]
        )

        gpu_price = (
            gpu_df.groupby("GPU_Brand")["Price_euros"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_gpu = px.bar(
            gpu_price,
            x="GPU_Brand",
            y="Price_euros",
            title="Average Laptop Price by GPU Brand",
            labels={
                "GPU_Brand": "GPU Brand",
                "Price_euros": "Average Price (€)"
            },
            text_auto=".0f"
        )

        st.plotly_chart(
            fig_gpu,
            use_container_width=True
        )

        # Storage analysis

        storage_df = filtered_df.copy()

        storage_df["Storage_Type"] = storage_df[
            "Memory"
        ].apply(
            lambda x:
                "SSD" if "SSD" in x.upper()
                else "HDD" if "HDD" in x.upper()
                else "Other"
        )

        storage_price = (
            storage_df.groupby("Storage_Type")["Price_euros"]
            .mean()
            .reset_index()
        )

        fig_storage = px.bar(
            storage_price,
            x="Storage_Type",
            y="Price_euros",
            title="Average Laptop Price by Storage Type",
            labels={
                "Storage_Type": "Storage Type",
                "Price_euros": "Average Price (€)"
            },
            text_auto=".0f"
        )

        st.plotly_chart(
            fig_storage,
            use_container_width=True
        )

        # Price distribution

        fig_price = px.histogram(
            filtered_df,
            x="Price_euros",
            nbins=30,
            title="Distribution of Laptop Prices",
            labels={
                "Price_euros": "Laptop Price (€)"
            }
        )

        st.plotly_chart(
            fig_price,
            use_container_width=True

        )

    else:

        st.warning(
            "Please select at least one brand."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🧠 Model Performance":

    st.title("🧠 Model Performance")

    st.write(
        "Comparison of Machine Learning models used "
        "for laptop price prediction."
    )

    st.divider()

    performance = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],

        "MAE (€)": [
            259.25,
            166.43,
            205.14
        ],

        "RMSE (€)": [
            347.27,
            260.78,
            290.45
        ],

        "R² Score": [
            0.684,
            0.822,
            0.779
        ]

    })

    st.subheader("📊 Model Comparison")

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    fig_model = px.bar(
        performance,
        x="Model",
        y="R² Score",
        title="R² Score Comparison",
        text="R² Score"
    )

    fig_model.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )

    st.success(
        "🏆 Random Forest is the best-performing model "
        "with an R² score of approximately 0.822."
    )

    st.info(
        "📌 Random Forest was selected as the final "
        "TechPrice AI prediction model."
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About TechPrice AI")

    st.write(
        "TechPrice AI is a Machine Learning based "
        "laptop price prediction system."
    )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The objective is to predict the expected price "
        "of a laptop using its hardware specifications "
        "and compare it with the current selling price."
    )

    st.subheader("🧠 Machine Learning Model")

    st.write(
        "Random Forest Regressor"
    )

    st.subheader("📊 Model Performance")

    st.write(
        "R² Score: approximately 0.82"
    )

    st.write(
        "Cross-Validation Average R²: approximately 0.79"
    )

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Plotly
    - Streamlit
    - Joblib
    """)

    st.subheader("📌 Key Features")

    st.write("""
    - 🤖 Laptop price prediction
    - 💰 Deal evaluation
    - 📊 Price analysis
    - 🧠 Model performance comparison
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "💻 TechPrice AI | Laptop Price Prediction "
    "using Machine Learning"
)