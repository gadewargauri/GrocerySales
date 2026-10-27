import os
import glob
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Grocery Product Sales  Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# PROFESSIONAL NAVY BLUE THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0B1220, #142238, #101827);
    color: #EAF2FF;
}
[data-testid="stHeader"] {
    background: #0B1220;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111D32, #192844);
    border-right: 1px solid #304765;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #DCE8FA !important;
}
h1 {
    color: #65E6FF !important;
    font-weight: 850 !important;
}
h2 {
    color: #67B7FF !important;
    font-weight: 800 !important;
}
h3 {
    color: #65DCCB !important;
    font-weight: 750 !important;
}
p, label {
    color: #D5E1F2;
}
.hero {
    background: linear-gradient(110deg, #164E9A, #245B91, #176D83);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #418CA9;
    margin-bottom: 24px;
}
.hero h1 {
    color: white !important;
    font-size: 35px !important;
}
.hero p {
    color: #E6F3FF !important;
}
.tagline {
    color: #A5F3E5 !important;
    font-size: 17px;
    font-weight: 700;
}
div[data-testid="stForm"] {
    background: #17263D;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #344C6D;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1D304B, #17263D);
    border: 1px solid #405879;
    border-radius: 16px;
    padding: 16px;
}
[data-testid="stMetricLabel"] {
    color: #C1D1E9 !important;
}
[data-testid="stMetricValue"] {
    color: #72E7D1 !important;
    font-weight: 800;
}
.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #087FCE, #147F91);
    color: white !important;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 800;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #086EB4, #116A79);
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #22344E;
    border-color: #49617F;
}
[data-baseweb="select"] *,
[data-baseweb="input"] input {
    color: #F3F7FF !important;
}
[data-testid="stPlotlyChart"] {
    background: #17263D;
    padding: 8px;
    border-radius: 16px;
    border: 1px solid #344C6D;
}
.result-card {
    background: linear-gradient(115deg, #103C43, #173A5A, #164C58);
    border: 1px solid #38CBB5;
    padding: 25px;
    border-radius: 20px;
    margin: 12px 0 20px 0;
}
.result-card h2 {
    color: #75F0D4 !important;
}
.result-card p {
    color: #E2EEFF !important;
}
hr {
    border-color: #344C6D;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE SEARCH
# =========================================================

def find_file(names):
    for name in names:
        path = os.path.join(BASE_DIR, name)
        if os.path.isfile(path):
            return path

    for name in names:
        matches = glob.glob(
            os.path.join(BASE_DIR, "**", name),
            recursive=True
        )
        if matches:
            return matches[0]

    return None


model_path = find_file([
    "best_model.pkl",
    "best_model.joblib",
    "best_model (2).pkl",
    "best_model (1).pkl"
])

columns_path = find_file([
    "model_columns.pkl",
    "columns.pkl",
    "model_columns.joblib"
])

scaler_path = find_file([
    "scaler.pkl",
    "scaler.joblib",
    "scaler (2).pkl",
    "scaler (1).pkl"
])

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset(path):
    if path.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)

    data.columns = data.columns.astype(str).str.strip()
    return data


dataset = None
dataset_path = None

# Search for the user's exact CSV filename first.
exact_paths = glob.glob(
    os.path.join(BASE_DIR, "**", "K_class_sales_clean.csv"),
    recursive=True
)

for path in exact_paths:
    try:
        dataset = load_dataset(path)
        dataset_path = path
        break
    except Exception:
        continue

# If the exact filename is not found, search compatible datasets.
if dataset is None:
    possible_files = (
        glob.glob(os.path.join(BASE_DIR, "**", "*.csv"), recursive=True)
        + glob.glob(os.path.join(BASE_DIR, "**", "*.xlsx"), recursive=True)
        + glob.glob(os.path.join(BASE_DIR, "**", "*.xls"), recursive=True)
    )

    for path in possible_files:
        if os.path.basename(path).lower().startswith(
            ("sample", "submission")
        ):
            continue

        try:
            temp = load_dataset(path)
            normalized = {
                col.lower().strip()
                for col in temp.columns
            }

            if (
                "item_outlet_sales" in normalized
                and "item_mrp" in normalized
            ):
                dataset = temp
                dataset_path = path
                break
        except Exception:
            continue

# =========================================================
# LOAD MODEL, FEATURE COLUMNS AND SCALER
# =========================================================

@st.cache_resource
def load_model_files(model_file, columns_file, scaler_file):
    loaded_model = joblib.load(model_file)
    loaded_columns = joblib.load(columns_file)

    loaded_scaler = (
        joblib.load(scaler_file)
        if scaler_file
        else None
    )

    return loaded_model, loaded_columns, loaded_scaler


model = None
model_columns = None
scaler = None
model_error = None

if model_path and columns_path:
    try:
        model, model_columns, scaler = load_model_files(
            model_path,
            columns_path,
            scaler_path
        )
    except Exception as error:
        model_error = str(error)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("# 🛒 Grocery World")
    st.markdown("### ✨ SMART GROCERY INSIGHTS")
    st.divider()

    st.markdown("### 🧺 Quick Guide")
    st.markdown("📝 Enter grocery details")
    st.markdown("🚀 Click the prediction button")
    st.markdown("📊 View your prediction and charts")

    st.divider()

    st.markdown("### 🌟 Explore")
    st.markdown("🥫 Grocery categories")
    st.markdown("🏬 Outlet performance")
    st.markdown("📈 Sales patterns")

    if dataset is not None:
        st.success("")
    else:
        st.warning("BigMart CSV not found")

    if model is not None and model_columns is not None:
        st.success("")
    else:
        st.warning("Model files need checking")

    st.caption("EVERYDAY GROCERIES. SMARTER POSSIBILITIES.")

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🛒 Grocery Product Sales Prediction</h1>
    <p class="tagline">
        ✨ Little details. Smarter predictions. Bigger possibilities.
    </p>
    <p>
        Enter your grocery and outlet details to estimate item outlet sales.
    </p>
</div>
""", unsafe_allow_html=True)

if dataset is not None:
    st.caption(f"Dataset loaded: {os.path.basename(dataset_path)}")

# =========================================================
# DATASET VALIDATION
# =========================================================

if dataset is not None:
    column_lookup = {
        col.lower().strip(): col
        for col in dataset.columns
    }
else:
    column_lookup = {}


def get_col(name):
    return column_lookup.get(name.lower())


def get_options(column, fallback):
    actual_column = get_col(column)

    if dataset is not None and actual_column is not None:
        values = (
            dataset[actual_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            return sorted(values) + ["Custom..."]

    return fallback.copy()


fat_options = get_options(
    "Item_Fat_Content",
    ["Low Fat", "Regular", "LF", "reg", "Custom..."]
)

item_type_options = get_options(
    "Item_Type",
    [
        "Dairy", "Soft Drinks", "Meat",
        "Fruits and Vegetables", "Household",
        "Baking Goods", "Snack Foods", "Frozen Foods",
        "Breakfast", "Health and Hygiene", "Hard Drinks",
        "Canned", "Breads", "Starchy Foods",
        "Others", "Seafood", "Custom..."
    ]
)

outlet_size_options = get_options(
    "Outlet_Size",
    ["Small", "Medium", "High", "Custom..."]
)

location_options = get_options(
    "Outlet_Location_Type",
    ["Tier 1", "Tier 2", "Tier 3", "Custom..."]
)

outlet_type_options = get_options(
    "Outlet_Type",
    [
        "Supermarket Type1", "Supermarket Type2",
        "Supermarket Type3", "Grocery Store", "Custom..."
    ]
)

# =========================================================
# INPUT FORM
# =========================================================

st.markdown("## 🧺 Grocery Sales Prediction")
st.write(
    "Enter the details and click the button to see your prediction and charts."
)

with st.form("grocery_prediction_form", clear_on_submit=False):

    st.markdown("### 🛍️ Grocery Item Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        item_weight = st.number_input(
            "⚖️ Item Weight",
            min_value=0.0,
            max_value=100000.0,
            value=12.0,
            step=0.1
        )

    with c2:
        item_visibility = st.number_input(
            "👀 Item Visibility",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f"
        )

    with c3:
        item_mrp = st.number_input(
            "🏷️ Item MRP (₹)",
            min_value=0.0,
            max_value=1000000.0,
            value=150.0,
            step=5.0
        )

    c4, c5 = st.columns(2)

    with c4:
        fat_choice = st.selectbox(
            "🥛 Item Fat Content",
            fat_options
        )

        if fat_choice == "Custom...":
            item_fat = st.text_input(
                "Enter Fat Content",
                "Low Fat"
            )
        else:
            item_fat = fat_choice

    with c5:
        type_choice = st.selectbox(
            "🛍️ Item Type",
            item_type_options
        )

        if type_choice == "Custom...":
            item_type = st.text_input(
                "Enter Item Type",
                "Dairy"
            )
        else:
            item_type = type_choice

    st.divider()
    st.markdown("### 🏬 Outlet Details")

    c6, c7, c8 = st.columns(3)

    with c6:
        size_choice = st.selectbox(
            "🏪 Outlet Size",
            outlet_size_options
        )

        if size_choice == "Custom...":
            outlet_size = st.text_input(
                "Enter Outlet Size",
                "Medium"
            )
        else:
            outlet_size = size_choice

    with c7:
        location_choice = st.selectbox(
            "📍 Outlet Location Type",
            location_options
        )

        if location_choice == "Custom...":
            outlet_location = st.text_input(
                "Enter Outlet Location",
                "Tier 2"
            )
        else:
            outlet_location = location_choice

    with c8:
        default_year = 2000

        year_col = get_col("Outlet_Establishment_Year")

        if dataset is not None and year_col:
            years = pd.to_numeric(
                dataset[year_col],
                errors="coerce"
            ).dropna()

            if not years.empty:
                default_year = int(years.median())

        establishment_year = st.number_input(
            "📅 Outlet Establishment Year",
            min_value=1900,
            max_value=2100,
            value=default_year,
            step=1
        )

    outlet_type_choice = st.selectbox(
        "🏬 Outlet Type",
        outlet_type_options
    )

    if outlet_type_choice == "Custom...":
        outlet_type = st.text_input(
            "Enter Outlet Type",
            "Supermarket Type1"
        )
    else:
        outlet_type = outlet_type_choice

    submitted = st.form_submit_button(
        "🚀 Predict My Grocery Sales",
        use_container_width=True
    )

# =========================================================
# SESSION STATE
# =========================================================

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# =========================================================
# PREDICTION
# =========================================================

if submitted:
    st.session_state.show_dashboard = False
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_inputs", None)

    if dataset is None:
        st.error(
            "Dataset not found. Place K_class_sales_clean.csv "
            "in the project folder or a subfolder."
        )

    elif model is None or model_columns is None:
        st.error(
            "Model files were not found or could not be loaded. "
            "Ensure best_model.pkl and model_columns.pkl are beside app.py."
        )

        if model_error:
            st.caption(f"Model loading error: {model_error}")

    elif not all([
        str(item_fat).strip(),
        str(item_type).strip(),
        str(outlet_size).strip(),
        str(outlet_location).strip(),
        str(outlet_type).strip()
    ]):
        st.error("Please fill in all grocery and outlet details.")

    else:
        fat_mapping = {
            "LF": "Low Fat",
            "low fat": "Low Fat",
            "reg": "Regular"
        }

        normalized_fat = fat_mapping.get(
            str(item_fat).strip(),
            str(item_fat).strip()
        )

        input_values = {
            "Item_Weight": item_weight,
            "Item_Fat_Content": normalized_fat,
            "Item_Visibility": item_visibility,
            "Item_Type": str(item_type).strip(),
            "Item_MRP": item_mrp,
            "Outlet_Establishment_Year": int(establishment_year),
            "Outlet_Size": str(outlet_size).strip(),
            "Outlet_Location_Type": str(outlet_location).strip(),
            "Outlet_Type": str(outlet_type).strip()
        }

        try:
            input_data = pd.DataFrame([input_values])

            # Match the preprocessing used during model training.
            input_data = pd.get_dummies(
                input_data,
                drop_first=True
            )

            input_data = input_data.reindex(
                columns=list(model_columns),
                fill_value=0
            )

            if scaler is not None:
                prediction = float(
                    model.predict(scaler.transform(input_data))[0]
                )
            else:
                prediction = float(
                    model.predict(input_data)[0]
                )

            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid prediction.")

            st.session_state.prediction = prediction
            st.session_state.prediction_inputs = input_values
            st.session_state.show_dashboard = True

        except Exception as error:
            st.session_state.show_dashboard = False
            st.error(f"Prediction failed: {error}")
            st.info(
                "Check that your saved model, feature columns and scaler "
                "match the preprocessing used during training."
            )

# =========================================================
# DISPLAY RESULT AND CHARTS AFTER SUCCESSFUL PREDICTION
# =========================================================

if (
    st.session_state.show_dashboard
    and "prediction" in st.session_state
    and dataset is not None
):
    prediction = st.session_state.prediction
    input_values = st.session_state.prediction_inputs

    st.markdown("---")
    st.markdown("## 🎉 Your Prediction Is Ready!")

    st.markdown(f"""
    <div class="result-card">
        <h2>🛍️ Estimated Item Outlet Sales</h2>
        <h1 style="color:#75F0D4 !important;font-size:44px;">
            ₹{prediction:,.2f}
        </h1>
        <p>✨ Estimated sales based on the details you entered.</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    m1.metric("🏷️ Item MRP", f"₹{input_values['Item_MRP']:,.2f}")
    m2.metric("⚖️ Item Weight", f"{input_values['Item_Weight']:.2f}")
    m3.metric("👀 Item Visibility", f"{input_values['Item_Visibility']:.3f}")

    st.markdown("### 🧾 Your Grocery Snapshot")

    snapshot = pd.DataFrame({
        "Feature": list(input_values.keys()) + ["Predicted Item Outlet Sales"],
        "Selected Value": [
            str(value) for value in input_values.values()
        ] + [f"₹{prediction:,.2f}"]
    })

    st.dataframe(
        snapshot,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # CHART SETTINGS
    # =====================================================

    COLORS = [
        "#00D4FF", "#FF6384", "#36E2B3", "#FFC857",
        "#A78BFA", "#FF8C42", "#4D96FF", "#F15BB5",
        "#B8F35B", "#00BBF9", "#FEE440", "#9B5DE5"
    ]

    def chart_style(fig, title, x_title=None, y_title=None, height=420):
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color="#65E6FF")
            ),
            template="plotly_dark",
            paper_bgcolor="#17263D",
            plot_bgcolor="#17263D",
            font=dict(color="#EAF2FF", size=12),
            height=height,
            margin=dict(l=35, r=25, t=75, b=55),
            xaxis_title=x_title,
            yaxis_title=y_title,
            legend_title_text="Category"
        )

        fig.update_xaxes(gridcolor="#34445D")
        fig.update_yaxes(gridcolor="#34445D")

        return fig

    df = dataset.copy()

    numeric_columns = [
        "Item_Weight",
        "Item_Visibility",
        "Item_MRP",
        "Item_Outlet_Sales",
        "Outlet_Establishment_Year",
        "Profit"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    st.markdown("---")
    st.markdown("## 🌈 Colorful Data Visualization Gallery")
    st.write(
        "Explore grocery categories, outlet performance, "
        "sales patterns and product relationships."
    )

    st.markdown("### 📌 Dataset at a Glance")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("📦 Total Records", f"{len(df):,}")
    k2.metric("🧾 Total Features", f"{len(df.columns):,}")

    if "Item_Outlet_Sales" in df.columns:
        k3.metric(
            "💰 Total Sales",
            f"₹{df['Item_Outlet_Sales'].sum():,.0f}"
        )
        k4.metric(
            "🛍️ Average Sales",
            f"₹{df['Item_Outlet_Sales'].mean():,.2f}"
        )

    # =====================================================
    # 1. PIE CHART
    # =====================================================

    st.markdown("---")
    st.markdown("### 🥧 1. Item Fat Content — Pie Chart")

    if "Item_Fat_Content" in df.columns:
        pie_data = (
            df["Item_Fat_Content"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Fat Content")
            .reset_index(name="Count")
        )

        fig = px.pie(
            pie_data,
            names="Fat Content",
            values="Count",
            color_discrete_sequence=COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        chart_style(fig, "🥧 Grocery Fat Content Composition")
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 2. DOUGHNUT CHART
    # =====================================================

    st.markdown("### 🍩 2. Outlet Size — Doughnut Chart")

    if "Outlet_Size" in df.columns:
        donut_data = (
            df["Outlet_Size"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Outlet Size")
            .reset_index(name="Count")
        )

        fig = px.pie(
            donut_data,
            names="Outlet Size",
            values="Count",
            hole=0.55,
            color_discrete_sequence=COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        chart_style(fig, "🍩 Outlet Size Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 3. AVERAGE SALES BY ITEM TYPE
    # =====================================================

    st.markdown("### 📊 3. Average Sales by Item Type")

    if {"Item_Type", "Item_Outlet_Sales"}.issubset(df.columns):
        temp = (
            df.dropna(subset=["Item_Type", "Item_Outlet_Sales"])
            .groupby("Item_Type", as_index=False)["Item_Outlet_Sales"]
            .mean()
            .sort_values("Item_Outlet_Sales", ascending=False)
        )

        fig = px.bar(
            temp,
            x="Item_Type",
            y="Item_Outlet_Sales",
            color="Item_Type",
            text_auto=".2s",
            color_discrete_sequence=COLORS
        )

        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-35)

        chart_style(
            fig,
            "📊 Average Sales by Grocery Category",
            "Item Type",
            "Average Sales (₹)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 4. SALES BY OUTLET TYPE
    # =====================================================

    st.markdown("### 🏬 4. Sales by Outlet Type")

    if {"Outlet_Type", "Item_Outlet_Sales"}.issubset(df.columns):
        temp = (
            df.dropna(subset=["Outlet_Type", "Item_Outlet_Sales"])
            .groupby("Outlet_Type", as_index=False)["Item_Outlet_Sales"]
            .sum()
            .sort_values("Item_Outlet_Sales", ascending=True)
        )

        fig = px.bar(
            temp,
            x="Item_Outlet_Sales",
            y="Outlet_Type",
            orientation="h",
            color="Outlet_Type",
            text_auto=".2s",
            color_discrete_sequence=COLORS
        )

        chart_style(
            fig,
            "🏬 Total Sales by Outlet Type",
            "Total Sales (₹)",
            "Outlet Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 5. LINE CHART
    # =====================================================

    st.markdown("### 📈 5. Sales Trend — Line Chart")

    if {
        "Outlet_Establishment_Year",
        "Item_Outlet_Sales"
    }.issubset(df.columns):

        temp = (
            df.dropna(subset=[
                "Outlet_Establishment_Year",
                "Item_Outlet_Sales"
            ])
            .groupby(
                "Outlet_Establishment_Year",
                as_index=False
            )["Item_Outlet_Sales"]
            .mean()
            .sort_values("Outlet_Establishment_Year")
        )

        fig = px.line(
            temp,
            x="Outlet_Establishment_Year",
            y="Item_Outlet_Sales",
            markers=True,
            color_discrete_sequence=["#00D4FF"]
        )

        fig.update_traces(line_width=4, marker_size=9)

        chart_style(
            fig,
            "📈 Average Sales by Establishment Year",
            "Establishment Year",
            "Average Sales (₹)"
        )

        st.plotly_chart(fig, use_container_width=True)

    
    

    
    
    
    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.markdown("---")
    st.markdown("### 🧾 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🔢 Numerical Summary")

    numeric_data = df.select_dtypes(include=np.number)

    if not numeric_data.empty:
        st.dataframe(
            numeric_data.describe().round(2),
            use_container_width=True
        )

    st.caption(
        "Note: Sales and other charts summarize your dataset. "
        
    )

elif submitted and dataset is None:
    st.warning(
        "The dataset was not found. Place K_class_sales_clean.csv "
        "in the project folder or one of its subfolders."
    )

else:
    st.info(
        "📝 Enter the grocery details above and click "
        "**Predict My Grocery Sales** to reveal your prediction "
        "and visualization dashboard."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    padding:22px;
    background:linear-gradient(110deg,#164E9A,#245B91,#176D83);
    border:1px solid #418CA9;
    border-radius:18px;
">
    <div style="font-size:23px;font-weight:850;color:white;">
        🛒 Grocery Price Prediction
    </div>
    <div style="font-size:15px;margin-top:8px;color:#DDEBFF;">
        ✨ Everyday groceries. Smarter possibilities. Brighter decisions.
    </div>
</div>
""", unsafe_allow_html=True)
