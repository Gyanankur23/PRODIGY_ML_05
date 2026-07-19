# Food Recognition & Calorie Estimator

A machine learning application that recognizes food items from images and estimates their calorie content. Built with TensorFlow and deployed using Streamlit.

## Features

- **Food Recognition**: Deep learning model using MobileNetV2 for image classification
- **Calorie Estimation**: Estimates calories based on recognized food items
- **Indian Food Database**: Calorie mapping for 255+ Indian dishes
- **Streamlit Deployment**: Easy-to-use web interface
- **Custom Training**: Train on your own dataset or use Food-101

## Project Structure

```
PRODIGY_ML_05/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── create_calorie_dataset.py   # Calorie mapping creation
├── prepare_sample_data.py      # Sample data structure setup
├── requirements.txt            # Python dependencies
├── indian_food.csv             # Indian food dataset
├── food_calorie_map.csv        # Generated calorie mapping
└── README.md                   # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Calorie Mapping

The calorie mapping is already generated, but if you need to regenerate it:

```bash
python create_calorie_dataset.py
```

This creates `food_calorie_map.csv` with calorie estimates for Indian foods.

### 3. Prepare Training Data

#### Option A: Use Food-101 Dataset

1. Download the Food-101 dataset from Kaggle: https://www.kaggle.com/dansbecker/food-101
2. Extract and place the `images` folder in the project directory as `food-101/images`
3. Run training:

```bash
python train_model.py
```

#### Option B: Use Custom Sample Data

1. Create sample data structure:

```bash
python prepare_sample_data.py
```

2. Add food images to each category folder in `sample_food_data/`
3. Each category should have at least 10-20 images
4. Run training:

```bash
python train_model.py
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Open the Streamlit app in your browser
2. Upload a food image (JPG, JPEG, or PNG)
3. Click "Analyze Food"
4. View the recognized food item and estimated calories

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: GlobalAveragePooling2D, Dropout, Dense layers
- **Input Size**: 224x224 RGB images
- **Output**: Softmax classification for food categories

## Training Configuration

- **Batch Size**: 32
- **Epochs**: 10
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Data Augmentation**: Rotation, shift, horizontal flip

## Calorie Estimation

Calories are estimated based on:
- Ingredient analysis from the Indian food database
- Standard serving sizes
- Nutritional values per ingredient

## Files Generated After Training

- `food_recognition_model.h5` - Trained Keras model
- `class_indices.json` - Class label mapping

## Notes

- The model requires TensorFlow 2.13.0
- For best results, use high-quality food images
- Calorie estimates are approximate and based on standard recipes
- The app works in placeholder mode if no trained model is found

## Troubleshooting

**Model not found error**: Run `train_model.py` to train the model first.

**Out of memory during training**: Reduce `BATCH_SIZE` in `train_model.py`.

**Poor prediction accuracy**: Increase training epochs or add more training data.

## License

This project is for educational purposes.
