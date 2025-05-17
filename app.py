import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime, timedelta
import io
import time
import random

# Configure the app
st.set_page_config(
    page_title="Multi-Themed Financial ML Application",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme persistence
if 'theme' not in st.session_state:
    st.session_state.theme = "welcome"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "AAPL"
if 'selected_period' not in st.session_state:
    st.session_state.selected_period = "1y"
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Function to set theme
def set_theme(theme):
    st.session_state.theme = theme
    st.rerun()

# Function to load the sample data when user doesn't upload one
def load_sample_data():
    # Create a sample financial dataset
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.normal(100, 10, 500).cumsum(),
        'High': np.random.normal(105, 10, 500).cumsum(),
        'Low': np.random.normal(95, 10, 500).cumsum(),
        'Close': np.random.normal(100, 10, 500).cumsum(),
        'Volume': np.random.randint(100000, 10000000, 500),
        'Revenue': np.random.normal(1000000, 100000, 500),
        'Expenses': np.random.normal(800000, 80000, 500),
        'Customer_Satisfaction': np.random.normal(4, 0.5, 500).clip(1, 5),
        'Marketing_Spend': np.random.normal(50000, 10000, 500),
        'Risk_Factor': np.random.choice([0, 1], size=500, p=[0.7, 0.3])
    })
    data['Profit'] = data['Revenue'] - data['Expenses']
    data['ROI'] = data['Profit'] / data['Marketing_Spend']
    return data

# Function to fetch stock data
def fetch_stock_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error(f"No data found for {ticker}. Using sample data instead.")
            return load_sample_data()
        return data.reset_index()
    except Exception as e:
        st.error(f"Error fetching stock data: {e}. Using sample data instead.")
        return load_sample_data()

# Function to encode the image to base64
def get_image_base64(url):
    return f"url({url})"

# Theme-specific background GIFs
zombie_bg = "https://media.giphy.com/media/xUPGcq0kyXkLQBKAKY/giphy.gif"  # Zombie market theme
futuristic_bg = "https://media.giphy.com/media/26uf2YTgF5upXUTm0/giphy.gif"  # Futuristic data visualization
got_bg = "https://media.giphy.com/media/l46Cy1rHbQ92uuLXa/giphy.gif"  # Game of Thrones themed visualization
gaming_bg = "https://media.giphy.com/media/2yqYbPakQKDFhNZbW9/giphy.gif"  # 8-bit gaming visualization
financial_data = "https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif"  # Financial data animation

# Theme-specific CSS
theme_css = {
    "welcome": f"""
        <style>
            .main .block-container {{
                padding-top: 1rem;
                background-color: #f0f2f6;
                background-image: {get_image_base64(financial_data)};
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: white;
            }}
            .welcome-card {{
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                color: white;
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                transition-duration: 0.4s;
            }}
            .stButton>button:hover {{
                background-color: #45a049;
            }}
            .theme-card {{
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 15px;
                min-height: 200px;
                transition: transform 0.3s ease;
                cursor: pointer;
                margin-bottom: 10px;
                color: white;
            }}
            .theme-card:hover {{
                transform: scale(1.05);
            }}
            .zombie-card {{
                border: 2px solid #8B0000;
            }}
            .futuristic-card {{
                border: 2px solid #00BFFF;
            }}
            .got-card {{
                border: 2px solid #FFD700;
            }}
            .gaming-card {{
                border: 2px solid #FF6347;
            }}
            .stHeader {{
                background-color: rgba(0, 0, 0, 0.7);
                padding: 10px;
            }}
        </style>
    """,
    "zombie": f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');
            .main .block-container {{
                padding-top: 1rem;
                background-color: #1a1a1a;
                background-image: {get_image_base64(zombie_bg)};
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: #8B0000;
            }}
            h1, h2, h3 {{
                font-family: 'Creepster', cursive;
                color: #8B0000;
                text-shadow: 2px 2px 4px #000;
            }}
            .stButton>button {{
                background-color: #8B0000;
                color: #E0E0E0;
                border: 2px solid #4d0000;
                font-family: 'Creepster', cursive;
            }}
            .stButton>button:hover {{
                background-color: #4d0000;
                color: #FFFFFF;
            }}
            .stTextInput>div>div>input, .stSelectbox>div>div>div, .stFileUploader {{
                background-color: #2d2d2d;
                color: #E0E0E0;
                border: 1px solid #8B0000;
            }}
            .stDataFrame {{
                background-color: rgba(45, 45, 45, 0.7);
                color: #E0E0E0;
                border: 1px solid #8B0000;
            }}
            .card {{
                background-color: rgba(45, 45, 45, 0.8);
                border: 2px solid #8B0000;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                color: #E0E0E0;
            }}
            .metrics-card {{
                background-color: rgba(139, 0, 0, 0.7);
                border-radius: 8px;
                padding: 10px;
                color: #E0E0E0;
            }}
            .stProgress > div > div > div > div {{
                background-color: #8B0000;
            }}
            .home-btn {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 99;
            }}
        </style>
    """,
    "futuristic": f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
            .main .block-container {{
                padding-top: 1rem;
                background-color: #0a0a20;
                background-image: {get_image_base64(futuristic_bg)};
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: #7DF9FF;
            }}
            h1, h2, h3 {{
                font-family: 'Orbitron', sans-serif;
                color: #00BFFF;
                text-shadow: 0 0 10px #00BFFF, 0 0 20px #00BFFF, 0 0 30px #00BFFF;
                letter-spacing: 2px;
            }}
            .stButton>button {{
                background-color: #3a0ca3;
                color: #7DF9FF;
                border: 2px solid #4cc9f0;
                font-family: 'Orbitron', sans-serif;
                border-radius: 25px;
            }}
            .stButton>button:hover {{
                background-color: #4cc9f0;
                color: #0a0a20;
                box-shadow: 0 0 15px #4cc9f0;
            }}
            .stTextInput>div>div>input, .stSelectbox>div>div>div, .stFileUploader {{
                background-color: rgba(10, 10, 32, 0.7);
                color: #7DF9FF;
                border: 1px solid #4cc9f0;
                border-radius: 25px;
            }}
            .stDataFrame {{
                background-color: rgba(10, 10, 32, 0.7);
                color: #7DF9FF;
                border: 1px solid #4cc9f0;
            }}
            .card {{
                background-color: rgba(10, 10, 32, 0.8);
                border: 2px solid #4cc9f0;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                color: #7DF9FF;
                box-shadow: 0 0 20px rgba(76, 201, 240, 0.5);
            }}
            .metrics-card {{
                background-color: rgba(58, 12, 163, 0.7);
                border-radius: 15px;
                padding: 10px;
                color: #7DF9FF;
                border: 1px solid #4cc9f0;
            }}
            .stProgress > div > div > div > div {{
                background-color: #4cc9f0;
            }}
            .home-btn {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 99;
            }}
        </style>
    """,
    "got": f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');
            .main .block-container {{
                padding-top: 1rem;
                background-color: #1a1a1a;
                background-image: {get_image_base64(got_bg)};
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: #D4AF37;
            }}
            h1, h2, h3 {{
                font-family: 'MedievalSharp', cursive;
                color: #D4AF37;
                text-shadow: 2px 2px 4px #000;
            }}
            .stButton>button {{
                background-color: #5C0000;
                color: #D4AF37;
                border: 2px solid #D4AF37;
                font-family: 'MedievalSharp', cursive;
            }}
            .stButton>button:hover {{
                background-color: #D4AF37;
                color: #5C0000;
            }}
            .stTextInput>div>div>input, .stSelectbox>div>div>div, .stFileUploader {{
                background-color: rgba(26, 26, 26, 0.7);
                color: #D4AF37;
                border: 1px solid #D4AF37;
            }}
            .stDataFrame {{
                background-color: rgba(26, 26, 26, 0.7);
                color: #D4AF37;
                border: 1px solid #D4AF37;
            }}
            .card {{
                background-color: rgba(26, 26, 26, 0.8);
                border: 2px solid #D4AF37;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                color: #D4AF37;
            }}
            .metrics-card {{
                background-color: rgba(92, 0, 0, 0.7);
                border-radius: 8px;
                padding: 10px;
                color: #D4AF37;
                border: 1px solid #D4AF37;
            }}
            .stProgress > div > div > div > div {{
                background-color: #D4AF37;
            }}
            .home-btn {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 99;
            }}
        </style>
    """,
    "gaming": f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
            .main .block-container {{
                padding-top: 1rem;
                background-color: #000080;
                background-image: {get_image_base64(gaming_bg)};
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: #FFFF00;
            }}
            h1, h2, h3 {{
                font-family: 'Press Start 2P', cursive;
                color: #FF6347;
                text-shadow: 2px 2px 0px #000;
                letter-spacing: 1px;
            }}
            .stButton>button {{
                background-color: #FF6347;
                color: #000;
                border: 4px solid #000;
                font-family: 'Press Start 2P', cursive;
                font-size: 12px;
            }}
            .stButton>button:hover {{
                background-color: #FFFF00;
                color: #FF6347;
            }}
            .stTextInput>div>div>input, .stSelectbox>div>div>div, .stFileUploader {{
                background-color: rgba(0, 0, 128, 0.7);
                color: #FFFF00;
                border: 2px solid #FF6347;
            }}
            .stDataFrame {{
                background-color: rgba(0, 0, 128, 0.7);
                color: #FFFF00;
                border: 2px solid #FF6347;
            }}
            .card {{
                background-color: rgba(0, 0, 128, 0.8);
                border: 4px solid #FF6347;
                border-radius: 0px;
                padding: 20px;
                margin-bottom: 20px;
                color: #FFFF00;
            }}
            .metrics-card {{
                background-color: rgba(255, 99, 71, 0.7);
                border-radius: 0px;
                padding: 10px;
                color: #FFFF00;
                border: 4px solid #000;
            }}
            .stProgress > div > div > div > div {{
                background-color: #FF6347;
            }}
            .home-btn {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 99;
            }}
        </style>
    """
}

# Apply CSS based on selected theme
st.markdown(theme_css[st.session_state.theme], unsafe_allow_html=True)

# Zombie-themed functions - Linear Regression
def zombie_linear_regression(data):
    """Perform linear regression with zombie theme"""
    st.markdown("<div style='text-align: center;'><img src='https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif' width='200px' style='margin: 20px 0;'/></div>", unsafe_allow_html=True)
    st.markdown("<h1>🧟‍♂️ Zombie Survival Prediction - Linear Regression 🧟‍♀️</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>As the financial apocalypse looms, only the fittest stocks survive. Our Linear Regression model 
        predicts which stocks will outlast the zombie market crash and which will become financial undead.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display animated loading bar for "infection analysis"
    with st.spinner("Analyzing infection rates..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

    # Select features for the model
    st.markdown("<h3>🧠 Select Brain Features for Survival Prediction</h3>", unsafe_allow_html=True)

    # Get numerical columns for feature selection
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove the target column from features if it's present
    if 'Close' in numerical_cols:
        numerical_cols.remove('Close')

    features_col1, features_col2 = st.columns(2)
    with features_col1:
        target_col = st.selectbox(
            "Select Survival Indicator (Target Variable)",
            ['Close'] + numerical_cols,
            index=0
        )

    with features_col2:
        feature_cols = st.multiselect(
            "Select Infection Vectors (Feature Variables)",
            [col for col in numerical_cols if col != target_col],
            default=[col for col in ['Open', 'High', 'Low', 'Volume'] if col in numerical_cols and col != target_col][:2]
        )

    # Prepare the data
    if len(feature_cols) > 0:
        # Drop rows with NaN values
        data_clean = data.dropna(subset=feature_cols + [target_col])

        X = data_clean[feature_cols]
        y = data_clean[target_col]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        with st.spinner("Training the survival model..."):
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store results in session state
            st.session_state.model_results['zombie'] = {
                'model_type': 'Linear Regression',
                'features': feature_cols,
                'target': target_col,
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist(),
                'mse': mse,
                'r2': r2,
                'coefficients': {feature: coef for feature, coef in zip(feature_cols, model.coef_)}
            }

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            st.markdown("<h3>🔮 Survival Metrics</h3>", unsafe_allow_html=True)
            st.metric("Mean Squared Error (Lower is better)", f"{mse:.4f}")
            st.metric("R² Score (Higher is better)", f"{r2:.4f}")

            st.markdown("<h3>🧟‍♂️ Infection Coefficients</h3>", unsafe_allow_html=True)
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_
            })
            st.dataframe(coef_df)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>🧪 Survival Prediction vs Reality</h3>", unsafe_allow_html=True)

            # Create a dataframe for visualization
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            }).reset_index(drop=True)

            # Plot results
            fig = px.scatter(results_df, x='Actual', y='Predicted', title="Actual vs Predicted Values")
            fig.add_trace(go.Scatter(x=[results_df['Actual'].min(), results_df['Actual'].max()], 
                                     y=[results_df['Actual'].min(), results_df['Actual'].max()],
                                     mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))

            # Customize the plot to match the zombie theme
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8B0000'),
                xaxis=dict(gridcolor='#8B0000', gridwidth=0.5, zerolinecolor='#8B0000'),
                yaxis=dict(gridcolor='#8B0000', gridwidth=0.5, zerolinecolor='#8B0000')
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🔪 Survival Factor Importance</h3>", unsafe_allow_html=True)

        importances = abs(model.coef_)
        indices = np.argsort(importances)[::-1]

        # Create feature importance plot
        fig = px.bar(
            x=importances[indices],
            y=[feature_cols[i] for i in indices],
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            title="Feature Importance for Survival"
        )

        # Customize plot to match theme
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B0000'),
            xaxis=dict(gridcolor='#8B0000', gridwidth=0.5),
            yaxis=dict(gridcolor='#8B0000', gridwidth=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Prediction component
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🧠 Test Your Own Survival Scenario</h3>", unsafe_allow_html=True)

        # Create input fields for each feature
        input_values = {}
        for feature in feature_cols:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].mean())

            input_values[feature] = st.slider(
                f"{feature} value",
                min_val, max_val, default_val,
                step=(max_val - min_val) / 100
            )

        # Make a prediction with the input values
        prediction_input = pd.DataFrame([input_values])
        prediction = model.predict(prediction_input)[0]

        st.markdown(f"<h4>Survival Prediction: {prediction:.2f}</h4>", unsafe_allow_html=True)

        # Add a visual indicator for the prediction
        survival_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [data[target_col].min(), data[target_col].max()]},
                'bar': {'color': "#8B0000"},
                'steps': [
                    {'range': [data[target_col].min(), data[target_col].quantile(0.33)], 'color': "darkred"},
                    {'range': [data[target_col].quantile(0.33), data[target_col].quantile(0.66)], 'color': "firebrick"},
                    {'range': [data[target_col].quantile(0.66), data[target_col].max()], 'color': "indianred"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            },
            title={'text': "Survival Gauge"}
        ))

        survival_gauge.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B0000'),
            height=250
        )

        st.plotly_chart(survival_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download predictions
        prediction_download = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_test - y_pred
        })

        csv = prediction_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="zombie_predictions.csv" class="btn" style="background-color: #8B0000; color: white; padding: 8px 12px; text-decoration: none; border-radius: 4px; border: 2px solid #4d0000;">Download Survival Predictions CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.error("Please select at least one feature")

# Futuristic-themed functions - K-Means Clustering
def futuristic_kmeans_clustering(data):
    """Perform K-means clustering with futuristic theme"""
    st.markdown("<div style='text-align: center;'><img src='https://media.giphy.com/media/26uf2YTgF5upXUTm0/giphy.gif' width='200px' style='margin: 20px 0;'/></div>", unsafe_allow_html=True)
    st.markdown("<h1>🔮 Quantum Market Cluster Analysis 🚀</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>Our advanced K-Means algorithm scans the financial multiverse to identify hidden patterns and clusters
        that are invisible to standard market analysis. Discover which dimensional plane your investments belong to.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display animated loading bar for "quantum scanning"
    with st.spinner("Initializing quantum scanners..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

    # Select features for the model
    st.markdown("<h3>🔬 Select Dimensional Variables for Analysis</h3>", unsafe_allow_html=True)

    # Get numerical columns for feature selection
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        feature_cols = st.multiselect(
            "Select Features for Clustering",
            numerical_cols,
            default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
        )

    with col2:
        n_clusters = st.slider("Number of Quantum Clusters", 2, 10, 3)

    # Prepare the data
    if len(feature_cols) >= 2:  # Need at least 2 features for meaningful clustering
        # Drop rows with NaN values
        data_clean = data.dropna(subset=feature_cols)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_clean[feature_cols])

        # Create and train the model
        with st.spinner("Calculating quantum clusters..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            # Add cluster labels to the data
            data_clean['Cluster'] = clusters

            # Calculate cluster centers
            cluster_centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_),
                columns=feature_cols
            )

            # Store results in session state
            st.session_state.model_results['futuristic'] = {
                'model_type': 'K-Means Clustering',
                'features': feature_cols,
                'n_clusters': n_clusters,
                'cluster_centers': cluster_centers.to_dict(),
                'cluster_labels': clusters.tolist(),
                'inertia': kmeans.inertia_
            }

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            st.markdown("<h3>📊 Quantum Cluster Metrics</h3>", unsafe_allow_html=True)
            st.metric("Inertia (Lower is better)", f"{kmeans.inertia_:.2f}")
            st.metric("Number of Clusters", n_clusters)
            st.metric("Data Points Analyzed", len(data_clean))

            st.markdown("<h3>💫 Cluster Centers</h3>", unsafe_allow_html=True)
            st.dataframe(cluster_centers)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>🌌 Cluster Distribution</h3>", unsafe_allow_html=True)

            # Create a cluster distribution plot
            cluster_counts = data_clean['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']

            fig = px.bar(
                cluster_counts, 
                x='Cluster', 
                y='Count',
                title="Points per Cluster",
                color='Cluster',
                color_continuous_scale='plasma'
            )

            # Customize the plot to match the futuristic theme
            fig.update_layout(
                plot_bgcolor='rgba(10, 10, 32, 0.8)',
                paper_bgcolor='rgba(10, 10, 32, 0)',
                font=dict(color='#7DF9FF'),
                xaxis=dict(gridcolor='#4cc9f0', gridwidth=0.5, zerolinecolor='#4cc9f0'),
                yaxis=dict(gridcolor='#4cc9f0', gridwidth=0.5, zerolinecolor='#4cc9f0')
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3D Visualization if we have 3+ features
        if len(feature_cols) >= 3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>🌠 3D Quantum Cluster Visualization</h3>", unsafe_allow_html=True)

            viz_features = st.multiselect(
                "Select 3 Features for 3D Visualization", 
                feature_cols,
                default=feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
            )

            if len(viz_features) == 3:
                fig = px.scatter_3d(
                    data_clean, 
                    x=viz_features[0],
                    y=viz_features[1],
                    z=viz_features[2],
                    color='Cluster',
                    color_continuous_scale='plasma',
                    opacity=0.7,
                    title="3D Quantum Cluster Visualization"
                )

                # Add cluster centers
                for i in range(n_clusters):
                    fig.add_scatter3d(
                        x=[cluster_centers.iloc[i][viz_features[0]]],
                        y=[cluster_centers.iloc[i][viz_features[1]]],
                        z=[cluster_centers.iloc[i][viz_features[2]]],
                        mode='markers',
                        marker=dict(
                            color='white',
                            size=10,
                            symbol='diamond',
                            line=dict(color='white', width=2)
                        ),
                        name=f'Cluster {i} Center'
                    )

                # Customize to match the futuristic theme
                fig.update_layout(
                    plot_bgcolor='rgba(10, 10, 32, 0.8)',
                    paper_bgcolor='rgba(10, 10, 32, 0)',
                    font=dict(color='#7DF9FF'),
                    scene=dict(
                        xaxis=dict(gridcolor='#4cc9f0', backgroundcolor='rgba(10, 10, 32, 0.8)'),
                        yaxis=dict(gridcolor='#4cc9f0', backgroundcolor='rgba(10, 10, 32, 0.8)'),
                        zaxis=dict(gridcolor='#4cc9f0', backgroundcolor='rgba(10, 10, 32, 0.8)')
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select exactly 3 features for 3D visualization")

            st.markdown("</div>", unsafe_allow_html=True)

        # 2D Visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>✨ 2D Cluster Mapping</h3>", unsafe_allow_html=True)

        if len(feature_cols) >= 2:
            viz_features_2d = st.multiselect(
                "Select 2 Features for 2D Visualization", 
                feature_cols,
                default=feature_cols[:2] if len(feature_cols) >= 2 else feature_cols
            )

            if len(viz_features_2d) == 2:
                fig = px.scatter(
                    data_clean, 
                    x=viz_features_2d[0],
                    y=viz_features_2d[1],
                    color='Cluster',
                    color_continuous_scale='plasma',
                    title="2D Quantum Cluster Mapping"
                )

                # Add cluster centers
                for i in range(n_clusters):
                    fig.add_scatter(
                        x=[cluster_centers.iloc[i][viz_features_2d[0]]],
                        y=[cluster_centers.iloc[i][viz_features_2d[1]]],
                        mode='markers',
                        marker=dict(
                            color='white',
                            size=15,
                            symbol='diamond',
                            line=dict(color='white', width=2)
                        ),
                        name=f'Cluster {i} Center'
                    )

                # Customize to match the futuristic theme
                fig.update_layout(
                    plot_bgcolor='rgba(10, 10, 32, 0.8)',
                    paper_bgcolor='rgba(10, 10, 32, 0)',
                    font=dict(color='#7DF9FF'),
                    xaxis=dict(gridcolor='#4cc9f0', gridwidth=0.5, zerolinecolor='#4cc9f0'),
                    yaxis=dict(gridcolor='#4cc9f0', gridwidth=0.5, zerolinecolor='#4cc9f0')
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select exactly 2 features for 2D visualization")

        st.markdown("</div>", unsafe_allow_html=True)

        # Cluster profiles
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🧬 Quantum Cluster Profiles</h3>", unsafe_allow_html=True)

        # Calculate cluster statistics
        cluster_profiles = data_clean.groupby('Cluster')[feature_cols].mean().reset_index()

        # Create radar chart
        categories = feature_cols
        fig = go.Figure()

        for i in range(n_clusters):
            cluster_data = cluster_profiles[cluster_profiles['Cluster'] == i].iloc[0]
            values = [cluster_data[cat] for cat in categories]

            # Normalize values for the radar chart
            min_vals = data_clean[categories].min()
            max_vals = data_clean[categories].max()
            normalized_values = [(val - min_vals[cat]) / (max_vals[cat] - min_vals[cat]) for cat, val in zip(categories, values)]

            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=categories,
                fill='toself',
                name=f'Cluster {i}',
                line_color=px.colors.sequential.Plasma[i % len(px.colors.sequential.Plasma)]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            plot_bgcolor='rgba(10, 10, 32, 0)',
            paper_bgcolor='rgba(10, 10, 32, 0)',
            font=dict(color='#7DF9FF')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download cluster data
        cluster_data = data_clean[feature_cols + ['Cluster']]
        csv = cluster_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="quantum_clusters.csv" class="btn" style="background-color: #3a0ca3; color: #7DF9FF; padding: 8px 12px; text-decoration: none; border-radius: 25px; border: 2px solid #4cc9f0;">Download Quantum Cluster Data</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.error("Please select at least two features for meaningful clustering")

# Game of Thrones-themed functions - Logistic Regression
def got_logistic_regression(data):
    """Perform logistic regression with Game of Thrones theme"""
    st.markdown("<div style='text-align: center;'><img src='https://media.giphy.com/media/l46Cy1rHbQ92uuLXa/giphy.gif' width='200px' style='margin: 20px 0;'/></div>", unsafe_allow_html=True)
    st.markdown("<h1>⚔️ House of Markets: Battle for Profits 🐉</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>In the realm of finance, only the strongest houses survive. Our Logistic Regression model predicts 
        which financial houses will rise to power and which will fall by the sword.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display animated loading bar for "battle preparations"
    with st.spinner("Consulting the Red Priestess..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

    # For logistic regression, we need a binary target
    # If no binary column exists in the data, we'll create one based on a threshold
    binary_cols = data.select_dtypes(include=['bool', 'int64']).columns.tolist()
    binary_cols = [col for col in binary_cols if data[col].nunique() <= 2]

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        # Get all numeric columns as potential targets
        potential_targets = data.select_dtypes(include=['float64', 'int64', 'bool']).columns.tolist()
        
        if 'Risk_Factor' in data.columns:
            default_target = 'Risk_Factor'
        elif len(binary_cols) > 0:
            default_target = binary_cols[0]
        elif len(potential_targets) > 0:
            default_target = potential_targets[0]
        else:
            st.error("No suitable numeric columns found for target variable")
            return

        target_col = st.selectbox(
            "Select Battle Outcome (Target Variable)",
            [default_target] + [col for col in potential_targets if col != default_target],
            index=0
        )

    with col2:
        # If the target is not binary, we need to create a binary version
        if target_col not in binary_cols:
            # Check if data is categorical
            if data[target_col].dtype == 'object':
                # For categorical data, map unique values to 0 and 1
                unique_vals = data[target_col].unique()
                if len(unique_vals) == 2:
                    # Create mapping using first value as 0 and second as 1
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    data['BinaryTarget'] = data[target_col].map(mapping)
                    st.info(f"Mapped {unique_vals[0]} to 0 and {unique_vals[1]} to 1")
                else:
                    st.error("Target column must have exactly 2 unique values for binary classification")
                    return
            else:
                # For numeric data, use threshold
                threshold_options = ['Median', 'Mean', 'Custom']
                threshold_type = st.selectbox("Threshold Type for Victory", threshold_options)

                if threshold_type == 'Custom':
                    min_val = float(data[target_col].min())
                    max_val = float(data[target_col].max())
                    threshold = st.slider("Victory Threshold", min_val, max_val, (min_val + max_val) / 2)
                elif threshold_type == 'Mean':
                    threshold = float(data[target_col].mean())
                    st.info(f"Mean Threshold: {threshold:.2f}")
                else:  # Median
                    threshold = float(data[target_col].median())
                    st.info(f"Median Threshold: {threshold:.2f}")

                # Create binary target
                data['BinaryTarget'] = (data[target_col] > threshold).astype(int)
            
            target_col = 'BinaryTarget'

    # Select features
    st.markdown("<h3>🛡️ Select Battle Weapons (Features)</h3>", unsafe_allow_html=True)

    feature_cols = st.multiselect(
        "Select Features for Battle Prediction",
        [col for col in numerical_cols if col != target_col and col != 'BinaryTarget'],
        default=[col for col in ['Open', 'High', 'Low', 'Volume'] if col in numerical_cols and col != target_col and col != 'BinaryTarget'][:2]
    )

    # Model parameters
    col1, col2 = st.columns(2)

    with col1:
        c_param = st.selectbox(
            "Regularization Strength (Iron Throne Control)",
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            index=3
        )

    with col2:
        max_iter = st.slider("Maximum Battle Iterations", 100, 2000, 1000, 100)

    # Prepare the data
    if len(feature_cols) > 0:
        # Drop rows with NaN values
        data_clean = data.dropna(subset=feature_cols + [target_col])

        X = data_clean[feature_cols]
        y = data_clean[target_col]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train the model
        with st.spinner("The battle is raging..."):
            model = LogisticRegression(C=c_param, max_iter=max_iter, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Store results in session state
            st.session_state.model_results['got'] = {
                'model_type': 'Logistic Regression',
                'features': feature_cols,
                'target': target_col,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'actual': y_test.tolist(),
                'accuracy': accuracy,
                'coefficients': {feature: coef for feature, coef in zip(feature_cols, model.coef_[0])}
            }

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            st.markdown("<h3>🏆 Battle Metrics</h3>", unsafe_allow_html=True)
            st.metric("Battle Accuracy", f"{accuracy:.4f}")

            st.markdown("<h3>⚔️ Battle Coefficients</h3>", unsafe_allow_html=True)
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_[0]
            })
            st.dataframe(coef_df)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>🔮 Victory Probabilities</h3>", unsafe_allow_html=True)

            # Create a histogram of probabilities
            prob_df = pd.DataFrame({
                'Probability': y_pred_proba,
                'Actual': y_test
            })

            fig = px.histogram(prob_df, x='Probability', color='Actual', 
                              title="Victory Probability Distribution",
                              labels={'Probability': 'Victory Probability', 'count': 'Count'},
                              opacity=0.8, nbins=20)

            # Customize to match GoT theme
            fig.update_layout(
                plot_bgcolor='rgba(26, 26, 26, 0.8)',
                paper_bgcolor='rgba(26, 26, 26, 0)',
                font=dict(color='#D4AF37'),
                xaxis=dict(gridcolor='#D4AF37', gridwidth=0.5, zerolinecolor='#D4AF37'),
                yaxis=dict(gridcolor='#D4AF37', gridwidth=0.5, zerolinecolor='#D4AF37'),
                legend_title="Actual Outcome"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Create a confusion matrix
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🗡️ Battle Confusion Matrix</h3>", unsafe_allow_html=True)

        cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

        # Create a figure with a custom confusion matrix visualization
        fig = go.Figure(data=go.Heatmap(
            z=cm.values,
            x=['Defeat (0)', 'Victory (1)'],
            y=['Defeat (0)', 'Victory (1)'],
            text=cm.values,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale=[[0, '#5C0000'], [1, '#D4AF37']]
        ))

        fig.update_layout(
            plot_bgcolor='rgba(26, 26, 26, 0.8)',
            paper_bgcolor='rgba(26, 26, 26, 0)',
            font=dict(color='#D4AF37')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🛡️ House Banners (Feature Importance)</h3>", unsafe_allow_html=True)

        importances = abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]

        fig = px.bar(
            x=importances[indices],
            y=[feature_cols[i] for i in indices],
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            title="House Banner Strength"
        )

        # Add house sigils (just for fun)
        house_sigils = ["🦁", "🐺", "🐉", "🦌", "🐙", "🌹", "🦑", "☀️", "🔥"]
        for i, importance in enumerate(importances[indices]):
            if i < len(house_sigils):
                fig.add_annotation(
                    x=importance,
                    y=feature_cols[indices[i]],
                    text=house_sigils[i],
                    showarrow=False,
                    xshift=10,
                    font=dict(size=20)
                )

        # Customize to match GoT theme
        fig.update_layout(
            plot_bgcolor='rgba(26, 26, 26, 0.8)',
            paper_bgcolor='rgba(26, 26, 26, 0)',
            font=dict(color='#D4AF37'),
            xaxis=dict(gridcolor='#D4AF37', gridwidth=0.5),
            yaxis=dict(gridcolor='#D4AF37', gridwidth=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Prediction component
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🏰 Test Your House's Strength</h3>", unsafe_allow_html=True)

        # Create input fields for each feature
        input_values = {}
        for feature in feature_cols:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].mean())

            input_values[feature] = st.slider(
                f"{feature} value",
                min_val, max_val, default_val,
                step=(max_val - min_val) / 100
            )

        # Make a prediction with the input values
        prediction_input = pd.DataFrame([input_values])
        prediction_input_scaled = scaler.transform(prediction_input)
        prediction = model.predict(prediction_input_scaled)[0]
        prediction_proba = model.predict_proba(prediction_input_scaled)[0][1]

        # Display prediction with GoT flair
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.markdown("<h4>⚔️ Prediction: VICTORY</h4>", unsafe_allow_html=True)
                st.markdown("Your house stands strong. The throne shall be yours!")
            else:
                st.markdown("<h4>☠️ Prediction: DEFEAT</h4>", unsafe_allow_html=True)
                st.markdown("The battle is lost. Winter has come for your house.")

        with col2:
            # Create a gauge chart for victory probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#D4AF37"},
                    'bar': {'color': "#D4AF37"},
                    'steps': [
                        {'range': [0, 33], 'color': "#5C0000"},
                        {'range': [33, 66], 'color': "#8B4513"},
                        {'range': [66, 100], 'color': "#DAA520"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_proba * 100
                    }
                },
                title={'text': "Victory Probability (%)"}
            ))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#D4AF37'),
                height=250
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Download predictions
        prediction_download = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Probability': y_pred_proba
        })

        csv = prediction_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="battle_predictions.csv" class="btn" style="background-color: #5C0000; color: #D4AF37; padding: 8px 12px; text-decoration: none; border-radius: 4px; border: 2px solid #D4AF37;">Download Battle Scrolls (Predictions CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.error("Please select at least one feature for battle")

# Gaming-themed functions - Random Forest
def gaming_random_forest(data):
    """Perform random forest regression with gaming theme"""
    st.markdown("<div style='text-align: center;'><img src='https://media.giphy.com/media/2yqYbPakQKDFhNZbW9/giphy.gif' width='200px' style='margin: 20px 0;'/></div>", unsafe_allow_html=True)
    st.markdown("<h1>🎮 Stock Market High Score Predictor 🕹️</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>Level up your investment strategy with our Random Forest Predictor! This powerful algorithm analyzes 
        market patterns like a pro gamer to predict future high scores (stock prices)!</p>
    </div>
    """, unsafe_allow_html=True)

    # Display animated loading bar for "game loading"
    with st.spinner("Loading game assets..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

    # Select features for the model
    st.markdown("<h3>🎲 Select Game Parameters</h3>", unsafe_allow_html=True)

    # Get numerical columns for feature selection
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Remove the target column from features if it's present
    if 'Close' in numerical_cols:
        numerical_cols.remove('Close')

    features_col1, features_col2 = st.columns(2)
    with features_col1:
        target_col = st.selectbox(
            "Select High Score (Target Variable)",
            ['Close'] + numerical_cols,
            index=0
        )

    with features_col2:
        feature_cols = st.multiselect(
            "Select Power-Ups (Feature Variables)",
            [col for col in numerical_cols if col != target_col],
            default=[col for col in ['Open', 'High', 'Low', 'Volume'] if col in numerical_cols and col != target_col][:2]
        )

    # Model parameters
    col1, col2 = st.columns(2)

    with col1:
        n_estimators = st.slider("Number of Trees (Game Levels)", 10, 200, 100, 10)

    with col2:
        max_depth = st.slider("Maximum Depth (Difficulty Level)", 2, 20, 10)

    # Prepare the data
    if len(feature_cols) > 0:
        # Drop rows with NaN values
        data_clean = data.dropna(subset=feature_cols + [target_col])

        X = data_clean[feature_cols]
        y = data_clean[target_col]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        with st.spinner("Training the AI boss..."):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store results in session state
            st.session_state.model_results['gaming'] = {
                'model_type': 'Random Forest',
                'features': feature_cols,
                'target': target_col,
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist(),
                'mse': mse,
                'r2': r2,
                'feature_importance': {feature: importance for feature, importance in zip(feature_cols, model.feature_importances_)}
            }

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            st.markdown("<h3>🎯 Game Stats</h3>", unsafe_allow_html=True)
            st.metric("Mean Squared Error (Lower is better)", f"{mse:.4f}")
            st.metric("R² Score (Higher is better)", f"{r2:.4f}")
            st.metric("Number of Trees", n_estimators)
            st.metric("Max Depth", max_depth)

            # Easter egg
            if random.random() < 0.3:  # 30% chance of showing easter egg
                st.markdown("🥚 Easter Egg: You found a hidden power-up! +10 Prediction Points! 🥚")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>📈 High Score Prediction</h3>", unsafe_allow_html=True)

            # Create a dataframe for visualization
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            }).reset_index(drop=True)

            # Plot results
            fig = px.scatter(results_df, x='Actual', y='Predicted', title="Actual vs Predicted High Scores")
            fig.add_trace(go.Scatter(x=[results_df['Actual'].min(), results_df['Actual'].max()], 
                                     y=[results_df['Actual'].min(), results_df['Actual'].max()],
                                     mode='lines', name='Perfect Score', line=dict(color='white', dash='dash')))

            # Customize the plot to match the gaming theme
            fig.update_layout(
                plot_bgcolor='rgba(0, 0, 128, 0.8)',
                paper_bgcolor='rgba(0, 0, 128, 0)',
                font=dict(color='#FFFF00'),
                xaxis=dict(gridcolor='#FF6347', gridwidth=0.5, zerolinecolor='#FF6347'),
                yaxis=dict(gridcolor='#FF6347', gridwidth=0.5, zerolinecolor='#FF6347')
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🔄 Power-Up Importance</h3>", unsafe_allow_html=True)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Create feature importance plot
        fig = px.bar(
            x=importances[indices],
            y=[feature_cols[i] for i in indices],
            orientation='h',
            labels={'x': 'Importance', 'y': 'Power-Up'},
            title="Power-Up Importance"
        )

        # Add pixel art style icons
        power_up_icons = ["🍄", "⭐", "🔥", "💰", "🔶", "💥", "🔋", "🌟", "🎁"]
        for i, importance in enumerate(importances[indices]):
            if i < len(power_up_icons):
                fig.add_annotation(
                    x=importance,
                    y=feature_cols[indices[i]],
                    text=power_up_icons[i],
                    showarrow=False,
                    xshift=10,
                    font=dict(size=20)
                )

        # Customize plot to match theme
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 128, 0.8)',
            paper_bgcolor='rgba(0, 0, 128, 0)',
            font=dict(color='#FFFF00'),
            xaxis=dict(gridcolor='#FF6347', gridwidth=0.5),
            yaxis=dict(gridcolor='#FF6347', gridwidth=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Plot residuals
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📊 Score Difference Analysis</h3>", unsafe_allow_html=True)

        residuals = y_test - y_pred
        residual_df = pd.DataFrame({
            'Predicted': y_pred,
            'Residuals': residuals
        })

        fig = px.scatter(
            residual_df, 
            x='Predicted', 
            y='Residuals',
            title="Residual Plot (Prediction Errors)"
        )

        fig.add_hline(y=0, line_dash="dash", line_color="white")

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 128, 0.8)',
            paper_bgcolor='rgba(0, 0, 128, 0)',
            font=dict(color='#FFFF00'),
            xaxis=dict(gridcolor='#FF6347', gridwidth=0.5),
            yaxis=dict(gridcolor='#FF6347', gridwidth=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Residual histogram
        fig = px.histogram(
            residual_df, 
            x='Residuals',
            title="Distribution of Prediction Errors",
            nbins=30,
            color_discrete_sequence=['#FF6347']
        )

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 128, 0.8)',
            paper_bgcolor='rgba(0, 0, 128, 0)',
            font=dict(color='#FFFF00'),
            xaxis=dict(gridcolor='#FF6347', gridwidth=0.5),
            yaxis=dict(gridcolor='#FF6347', gridwidth=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Prediction component
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🎮 Play Your Own Game</h3>", unsafe_allow_html=True)

        # Create input fields for each feature
        input_values = {}
        for feature in feature_cols:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].mean())

            input_values[feature] = st.slider(
                f"{feature} value",
                min_val, max_val, default_val,
                step=(max_val - min_val) / 100
            )

        # Make a prediction with the input values
        prediction_input = pd.DataFrame([input_values])
        prediction = model.predict(prediction_input)[0]

        st.markdown(f"<h4>Predicted High Score: {prediction:.2f}</h4>", unsafe_allow_html=True)

        # Add a visual indicator for the prediction
        score_gauge = go.Figure(go.Indicator(
mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [data[target_col].min(), data[target_col].max()]},
                'bar': {'color': "#FF6347"},
                'steps': [
                    {'range': [data[target_col].min(), data[target_col].quantile(0.33)], 'color': "#000080"},
                    {'range': [data[target_col].quantile(0.33), data[target_col].quantile(0.66)], 'color': "#0000FF"},
                    {'range': [data[target_col].quantile(0.66), data[target_col].max()], 'color': "#4169E1"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            },
            title={'text': "High Score Meter"}
        ))

        score_gauge.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFF00'),
            height=250
        )

        st.plotly_chart(score_gauge, use_container_width=True)

        # Add a pixelated progress bar
        st.markdown("""
        <div style="border: 4px solid black; padding: 2px; background-color: #000;">
            <div style="height: 20px; width: {}%; background-color: #FF6347; position: relative;">
                <div style="position: absolute; left: 0; right: 0; text-align: center; color: white; text-shadow: 1px 1px 0px black; font-family: 'Press Start 2P', cursive;">
                    LOADING...
                </div>
            </div>
        </div>
        """.format(min(100, max(0, (prediction - data[target_col].min()) / (data[target_col].max() - data[target_col].min()) * 100))), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Download predictions
        prediction_download = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_test - y_pred
        })

        csv = prediction_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="game_predictions.csv" class="btn" style="background-color: #FF6347; color: black; padding: 8px 12px; text-decoration: none; border: 4px solid #000;">SAVE GAME DATA</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.error("Please select at least one power-up (feature)")

# Welcome page
def welcome_page():
    st.markdown("<h1 style='text-align: center;'>Multi-Themed Financial ML Application</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-card">
        <h2>Welcome to the Financial ML Explorer!</h2>
        <p>This application allows you to analyze financial data using different machine learning models, 
        each presented with a unique visual theme:</p>
        <ul>
            <li><strong>Zombie Theme:</strong> Linear Regression for stock price prediction</li>
            <li><strong>Futuristic Theme:</strong> K-Means Clustering for pattern detection</li>
            <li><strong>Game of Thrones Theme:</strong> Logistic Regression for classification</li>
            <li><strong>Gaming Theme:</strong> Random Forest for performance prediction</li>
        </ul>
        <p>Choose a theme below to get started!</p>
    </div>
    """, unsafe_allow_html=True)

    # Data upload section
    st.markdown("<div class='welcome-card'>", unsafe_allow_html=True)
    st.markdown("<h2>Step 1: Upload or Select Data</h2>", unsafe_allow_html=True)

    data_source = st.radio(
        "Choose a data source:",
        ["Upload CSV file", "Fetch stock data", "Use sample data"]
    )

    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns")

                # Display data preview
                st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.data = load_sample_data()
        else:
            if st.session_state.data is None:
                st.warning("No file uploaded. Using sample data.")
                st.session_state.data = load_sample_data()

    elif data_source == "Fetch stock data":
        col1, col2 = st.columns(2)

        with col1:
            ticker = st.text_input("Stock Symbol", st.session_state.selected_stock)

        with col2:
            period_options = {
                "1 month": "1mo",
                "3 months": "3mo",
                "6 months": "6mo",
                "1 year": "1y",
                "2 years": "2y",
                "5 years": "5y",
                "10 years": "10y",
                "Year to date": "ytd",
                "Max": "max"
            }
            period = st.selectbox(
                "Time Period",
                list(period_options.keys()),
                index=list(period_options.values()).index(st.session_state.selected_period)
            )

        if st.button("Fetch Stock Data"):
            # Update session state
            st.session_state.selected_stock = ticker
            st.session_state.selected_period = period_options[period]

            # Fetch the data
            with st.spinner(f"Fetching data for {ticker}..."):
                data = fetch_stock_data(ticker, period_options[period])
                st.session_state.data = data
                st.session_state.stock_data = data

            # Display data
            st.success(f"Successfully loaded data for {ticker} with {data.shape[0]} rows")

            # Display data preview
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(data.head())

            # Display stock chart
            if 'Date' in data.columns and 'Close' in data.columns:
                # Create a simple figure using matplotlib instead of plotly
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(data['Date'], data['Close'])
                ax.set_title(f"{ticker} Stock Price - {period}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price")
                ax.grid(True)
                fig.autofmt_xdate()  # Rotate date labels
                st.pyplot(fig)

    else:  # Use sample data
        if st.button("Load Sample Data") or st.session_state.data is None:
            st.session_state.data = load_sample_data()
            st.success("Sample data loaded successfully")

            # Display data preview
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(st.session_state.data.head())

    st.markdown("</div>", unsafe_allow_html=True)

    # Theme selection
    st.markdown("<div class='welcome-card'>", unsafe_allow_html=True)
    st.markdown("<h2>Step 2: Choose Your Theme</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="theme-card zombie-card" onclick="document.getElementById('zombie-btn').click()">
                <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="100%" style="margin-bottom: 10px; border-radius: 8px;"/>
                <h3>🧟‍♂️ Zombie Apocalypse</h3>
                <p>Survive the financial apocalypse with Linear Regression</p>
                <p><strong>Model:</strong> Linear Regression</p>
                <p><strong>Best for:</strong> Predicting continuous values like stock prices</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        zombie_btn = st.button("Choose Zombie Theme", key="zombie-btn", help="Use Linear Regression with a zombie apocalypse theme")
        if zombie_btn:
            set_theme("zombie")

        st.markdown(
            """
            <div class="theme-card got-card" onclick="document.getElementById('got-btn').click()">
                <img src="https://media.giphy.com/media/l46Cy1rHbQ92uuLXa/giphy.gif" width="100%" style="margin-bottom: 10px; border-radius: 8px;"/>
                <h3>⚔️ Game of Thrones</h3>
                <p>Battle for the financial throne with Logistic Regression</p>
                <p><strong>Model:</strong> Logistic Regression</p>
                <p><strong>Best for:</strong> Classification into categories (win/lose, buy/sell)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        got_btn = st.button("Choose Game of Thrones Theme", key="got-btn", help="Use Logistic Regression with a Game of Thrones theme")
        if got_btn:
            set_theme("got")

    with col2:
        st.markdown(
            """
            <div class="theme-card futuristic-card" onclick="document.getElementById('future-btn').click()">
                <img src="https://media.giphy.com/media/26uf2YTgF5upXUTm0/giphy.gif" width="100%" style="margin-bottom: 10px; border-radius: 8px;"/>
                <h3>🚀 Futuristic Tech</h3>
                <p>Discover quantum patterns with K-Means Clustering</p>
                <p><strong>Model:</strong> K-Means Clustering</p>
                <p><strong>Best for:</strong> Finding groups and patterns in data</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        future_btn = st.button("Choose Futuristic Theme", key="future-btn", help="Use K-Means Clustering with a futuristic theme")
        if future_btn:
            set_theme("futuristic")

        st.markdown(
            """
            <div class="theme-card gaming-card" onclick="document.getElementById('gaming-btn').click()">
                <img src="https://media.giphy.com/media/2yqYbPakQKDFhNZbW9/giphy.gif" width="100%" style="margin-bottom: 10px; border-radius: 8px;"/>
                <h3>🎮 Retro Gaming</h3>
                <p>Level up your predictions with Random Forest</p>
                <p><strong>Model:</strong> Random Forest</p>
                <p><strong>Best for:</strong> Complex predictions with multiple factors</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        gaming_btn = st.button("Choose Gaming Theme", key="gaming-btn", help="Use Random Forest with a retro gaming theme")
        if gaming_btn:
            set_theme("gaming")

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="welcome-card" style="text-align: center;">
        <h3>🔮 Ready to explore the financial multiverse? 🔮</h3>
        <p>Select a theme to start your journey!</p>
    </div>
    """, unsafe_allow_html=True)

# Add a home button to themed pages
def home_button():
    if st.button("🏠 Return to Home", key="home-btn"):
        set_theme("welcome")

# Main application logic
if st.session_state.theme == "welcome":
    welcome_page()
elif st.session_state.data is None:
    # If no data is loaded, go back to welcome page
    st.error("No data available. Please upload or select data first.")
    set_theme("welcome")
elif st.session_state.theme == "zombie":
    zombie_linear_regression(st.session_state.data)
    home_button()
elif st.session_state.theme == "futuristic":
    futuristic_kmeans_clustering(st.session_state.data)
    home_button()
elif st.session_state.theme == "got":
    got_logistic_regression(st.session_state.data)
    home_button()
elif st.session_state.theme == "gaming":
    gaming_random_forest(st.session_state.data)
    home_button()