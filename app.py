import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
import json
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Food Recognition & Calorie Estimator",
    page_icon="🍔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #000000;
    }
    .result-box h3 {
        color: #000000;
    }
    .result-box p {
        color: #000000;
    }
    .calorie-display {
        font-size: 2rem;
        font-weight: bold;
        color: #000000
    }
</style>
""", unsafe_allow_html=True)

# Load calorie mapping
@st.cache_data
def load_calorie_data():
    """Load the calorie mapping dataset"""
    try:
        calorie_df = pd.read_csv('food_calorie_map.csv')
        return calorie_df
    except FileNotFoundError:
        # Create a fallback calorie mapping
        data = {
            'name': ['pizza', 'burger', 'salad', 'pasta', 'rice', 'biryani', 'chicken', 'dal', 'naan', 'samosa'],
            'calories': [285, 295, 150, 220, 130, 350, 239, 150, 262, 260],
            'diet': ['vegetarian', 'non vegetarian', 'vegetarian', 'vegetarian', 'vegetarian', 'non vegetarian', 'non vegetarian', 'vegetarian', 'vegetarian', 'vegetarian'],
            'course': ['main course', 'main course', 'main course', 'main course', 'main course', 'main course', 'main course', 'main course', 'main course', 'snack']
        }
        return pd.DataFrame(data)

# Load model
@st.cache_resource
def load_model():
    """Load the trained food recognition model"""
    try:
        model = tf.keras.models.load_model('food_recognition_model.h5')
        
        # Load class indices
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        # Create reverse mapping (index to class name)
        class_names = {v: k for k, v in class_indices.items()}
        
        return model, class_names
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning("Trained model not found. Using placeholder predictions.")
        return None, None

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_food(model, img_array, class_names):
    """Make prediction on food image"""
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    predicted_class = class_names.get(predicted_class_idx, "Unknown")
    
    return predicted_class, confidence

def get_calorie_estimate(food_name, calorie_df):
    """Get calorie estimate for recognized food"""
    # Try exact match first
    exact_match = calorie_df[calorie_df['name'].str.lower() == food_name.lower()]
    if not exact_match.empty:
        row = exact_match.iloc[0]
        return int(row['calories']), row['diet'], row['course']
    
    # Try partial match
    partial_match = calorie_df[calorie_df['name'].str.contains(food_name.lower(), case=False, na=False)]
    if not partial_match.empty:
        row = partial_match.iloc[0]
        return int(row['calories']), row['diet'], row['course']
    
    # Default estimate
    return 250, 'unknown', 'unknown'

def main():
    # Header
    st.markdown('<h1 class="main-header">🍔 Food Recognition & Calorie Estimator</h1>', unsafe_allow_html=True)
    
    # Load data
    calorie_df = load_calorie_data()
    model, class_names = load_model()
    
    # Create columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Food Image")
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Add analyze button
            if st.button("Analyze Food", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    img_array = preprocess_image(img)
                    
                    if model is not None and class_names is not None:
                        # Make prediction
                        food_name, confidence = predict_food(model, img_array, class_names)
                        
                        # Get calorie estimate
                        calories, diet, course = get_calorie_estimate(food_name, calorie_df)
                        
                        # Display results
                        with col2:
                            st.subheader("Analysis Results")
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>🍽️ Recognized Food: {food_name.replace('_', ' ').title()}</h3>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                <p><strong>Diet Type:</strong> {diet.replace('_', ' ').title()}</p>
                                <p><strong>Course:</strong> {course.replace('_', ' ').title()}</p>
                                <p class="calorie-display">🔥 Estimated Calories: {calories} kcal</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Additional information
                            st.info("💡 Tip: Calorie estimates are based on standard serving sizes. Actual calories may vary based on portion size and preparation method.")
                    else:
                        # Placeholder prediction when model is not available
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Simulate prediction
                            sample_foods = ['pizza', 'burger', 'salad', 'pasta', 'rice']
                            food_name = sample_foods[np.random.randint(0, len(sample_foods))]
                            calories, diet, course = get_calorie_estimate(food_name, calorie_df)
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>🍽️ Recognized Food: {food_name.title()}</h3>
                                <p><strong>Confidence:</strong> 85.0%</p>
                                <p><strong>Diet Type:</strong> {diet.title()}</p>
                                <p><strong>Course:</strong> {course.title()}</p>
                                <p class="calorie-display">🔥 Estimated Calories: {calories} kcal</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.warning("⚠️ This is a placeholder prediction. Train the model using train_model.py for accurate food recognition.")
    
    with col2:
        if uploaded_file is None:
            st.subheader("How It Works")
            st.markdown("""
            1. **Upload Image**: Select a food image from your device
            2. **Analyze**: Click the analyze button to process the image
            3. **View Results**: See the recognized food and estimated calories
            
            The model uses deep learning to recognize food items from images and estimates calories based on nutritional databases.
            """)
            
            st.subheader("Supported Foods")
            st.write("The model can recognize various Indian and international dishes including:")
            
            # Display sample foods from calorie database
            sample_foods = calorie_df['name'].head(10).tolist()
            for food in sample_foods:
                st.write(f"• {food}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with TensorFlow, Streamlit & ❤️ for healthy eating</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
