import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Define mappings for categorical variables
DISTRIBUTOR_MAP = {
    "Warner Bros.": 0,
    "Universal": 1,
    "Paramount": 2,
    "Sony": 3,
    "Disney": 4,
    "Other": 5
}

MPAA_MAP = {
    "G": 0,
    "PG": 1,
    "PG-13": 2,
    "R": 3
}

# Set page config
st.set_page_config(
    page_title="Box Office Revenue Predictor",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Box Office Revenue Predictor")
st.write("Enter movie details to predict its domestic revenue")

# Create input form
col1, col2 = st.columns(2)

with col1:
    mpaa_rating = st.selectbox(
        "MPAA Rating",
        ["G", "PG", "PG-13", "R"],
        key="mpaa"
    )
    
    opening_theaters = st.number_input(
        "Number of Opening Theaters",
        min_value=1,
        max_value=5000,
        value=2000,
        key="theaters"
    )
    
    opening_revenue = st.number_input(
        "Opening Revenue ($)",
        min_value=0,
        value=1000000,
        key="opening_rev"
    )

with col2:
    release_days = st.number_input(
        "Release Days",
        min_value=1,
        max_value=365,
        value=90,
        key="days"
    )
    
    distributor = st.selectbox(
        "Distributor",
        list(DISTRIBUTOR_MAP.keys()),
        key="dist"
    )
    
    world_revenue = st.number_input(
        "World Revenue ($)",
        min_value=0,
        value=2000000,
        key="world_rev"
    )

# Get unique genres from feature names
genres = ['action', 'animation', 'comedy', 'drama', 'horror', 'thriller']
selected_genres = st.multiselect(
    "Select Genres", 
    genres,
    format_func=lambda x: x.title()
)

# Predict button
if st.button("Predict Revenue"):
    try:
        # Initialize input data with the correct order of features
        input_data = pd.DataFrame(columns=feature_names)
        input_data.loc[0] = 0  # Initialize with zeros
        
        # Fill in the values with proper encoding
        input_data.loc[0, 'MPAA'] = MPAA_MAP[mpaa_rating]
        input_data.loc[0, 'opening_theaters'] = np.log10(opening_theaters)
        input_data.loc[0, 'opening_revenue'] = np.log10(opening_revenue)
        input_data.loc[0, 'release_days'] = np.log10(release_days)
        input_data.loc[0, 'world_revenue'] = np.log10(world_revenue)
        input_data.loc[0, 'distributor'] = DISTRIBUTOR_MAP[distributor]
        
        # Set genre values
        for genre in genres:
            input_data.loc[0, genre] = 1 if genre in selected_genres else 0
        
        # Ensure column order matches training data
        input_data = input_data[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Convert log prediction back to actual dollars
        revenue = 10 ** prediction
        
        # Display prediction
        st.success(f"Predicted Domestic Revenue: ${revenue:,.2f}")
        
        # Display input summary with readable values
        st.write("### Input Summary")
        summary_data = input_data.iloc[0].copy()
        # Convert encoded values back to readable form
        summary_data['MPAA'] = [k for k, v in MPAA_MAP.items() if v == summary_data['MPAA']][0]
        summary_data['distributor'] = [k for k, v in DISTRIBUTOR_MAP.items() if v == summary_data['distributor']][0]
        
        # Create formatted summary
        summary = pd.DataFrame({
            'Feature': summary_data.index,
            'Value': [
                f"${10**v:,.2f}" if name in ['opening_revenue', 'world_revenue']
                else f"{10**v:,.0f}" if name in ['opening_theaters', 'release_days']
                else v
                for name, v in summary_data.items()
            ]
        })
        st.dataframe(summary)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Available features: " + ", ".join(feature_names))
        st.error("Input data types: \n" + str(input_data.dtypes))

# Add explanatory notes
st.markdown("---")
st.markdown("""
### Notes:
- Opening Revenue: The revenue earned during the opening weekend
- World Revenue: The total worldwide revenue expectation
- Release Days: Number of days the movie is planned to be in theaters
- Opening Theaters: Number of theaters showing the movie on opening weekend
""")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Box Office Revenue Prediction Project")