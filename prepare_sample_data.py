"""
Script to prepare sample data structure for training.
This creates a simple directory structure and provides instructions.
"""
import os
import shutil

def create_sample_structure():
    """Create sample data directory structure"""
    sample_dir = 'sample_food_data'
    
    # Remove existing directory if it exists
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    
    # Create main directory
    os.makedirs(sample_dir, exist_ok=True)
    
    # Define food categories (using common Indian foods from our dataset)
    categories = [
        'biryani',
        'butter_chicken',
        'dal_makhani',
        'naan',
        'paneer_tikka',
        'samosa',
        'dosa',
        'idli',
        'chicken_curry',
        'rice'
    ]
    
    # Create subdirectories for each category
    for category in categories:
        category_path = os.path.join(sample_dir, category)
        os.makedirs(category_path, exist_ok=True)
    
    print(f"Created sample data structure at: {sample_dir}")
    print(f"Categories created: {categories}")
    print("\nNext steps:")
    print("1. Download food images for each category")
    print("2. Place images in the respective category folders")
    print("3. Each category should have at least 10-20 images for training")
    print("4. Run train_model.py to train the model")
    
    return sample_dir, categories

if __name__ == '__main__':
    create_sample_structure()
