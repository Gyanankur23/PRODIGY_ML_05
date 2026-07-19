import pandas as pd
import numpy as np

# Load the Indian food dataset
df = pd.read_csv('indian_food.csv')

# Calorie estimates per ingredient (per 100g/100ml)
calorie_database = {
    # Grains and flours
    'maida': 364, 'flour': 364, 'wheat flour': 340, 'whole wheat flour': 340,
    'rice': 130, 'rice flour': 364, 'besan': 387, 'gram flour': 387,
    'semolina': 360, 'rava': 360, 'sooji': 360, 'corn flour': 370,
    'jowar flour': 349, 'pearl millet flour': 378, 'sorghum flour': 349,
    'black lentils': 347, 'urad dal': 347, 'moong dal': 347, 'chana dal': 381,
    'toor dal': 347, 'pigeon peas': 347, 'arhar dal': 347,
    
    # Dairy
    'milk': 61, 'condensed milk': 321, 'milk powder': 496, 'curd': 98,
    'yogurt': 98, 'cottage cheese': 265, 'paneer': 265, 'chhena': 265,
    'khoa': 420, 'mawa': 420, 'cream': 340, 'ghee': 900, 'clarified butter': 900,
    'butter': 717, 'cheese': 402,
    
    # Sweeteners
    'sugar': 387, 'jaggery': 383, 'honey': 304, 'powdered sugar': 387,
    
    # Fruits and vegetables
    'carrots': 41, 'potato': 77, 'potatoes': 77, 'cauliflower': 25,
    'ladies finger': 33, 'bhindi': 33, 'peas': 81, 'green peas': 81,
    'tomato': 18, 'tomatoes': 18, 'onion': 40, 'onions': 40,
    'cucumber': 16, 'spinach': 23, 'bottle gourd': 14, 'lauki': 14,
    'bitter gourd': 17, 'karela': 17, 'pumpkin': 26, 'coconut': 354,
    'banana': 89, 'cashews': 553, 'almonds': 579, 'pistachio': 560,
    'raisins': 299, 'dry fruits': 400, 'apricots': 48,
    
    # Proteins
    'chicken': 239, 'fish': 206, 'prawns': 99, 'lobster': 89,
    'beef': 250, 'pork': 242, 'mutton': 294, 'lamb': 294,
    'eggs': 155, 'egg': 155,
    
    # Oils and fats
    'oil': 884, 'vegetable oil': 884, 'olive oil': 884, 'mustard oil': 884,
    'coconut oil': 862, 'sesame oil': 884,
    
    # Spices and flavorings (negligible calories)
    'ginger': 80, 'garlic': 149, 'chilli': 40, 'chillies': 40,
    'turmeric': 354, 'cardamom': 311, 'saffron': 310, 'cinnamon': 247,
    'clove': 274, 'garam masala': 200, 'curry leaves': 50,
    'cumin': 375, 'mustard seeds': 508, 'fenugreek': 323,
    'black pepper': 255, 'red chili': 40,
    
    # Others
    'bread': 265, 'naan': 262, 'vermicelli': 375, 'sev': 500,
    'peanuts': 567, 'chikki': 500, 'papad': 300,
}

def estimate_calories(ingredients_str):
    """Estimate calories based on ingredients"""
    if pd.isna(ingredients_str) or ingredients_str == '-1':
        return 250  # Default estimate
    
    ingredients = ingredients_str.lower().split(', ')
    total_calories = 0
    ingredient_count = 0
    
    for ingredient in ingredients:
        # Try to find matching ingredient in database
        matched = False
        for key, calories in calorie_database.items():
            if key in ingredient:
                total_calories += calories
                ingredient_count += 1
                matched = True
                break
        
        if not matched:
            # Add base calories for unknown ingredients
            total_calories += 100
            ingredient_count += 1
    
    # Average calories per ingredient, then adjust for typical serving
    if ingredient_count > 0:
        avg_calories = total_calories / ingredient_count
        # Adjust for typical serving size (multiply by 2 for main dishes, 1.5 for snacks)
        estimated = avg_calories * 2
    else:
        estimated = 250
    
    return int(estimated)

# Apply calorie estimation
df['calories'] = df['ingredients'].apply(estimate_calories)

# Create a simplified food-to-calorie mapping
food_calorie_map = df[['name', 'calories', 'diet', 'course']].copy()
food_calorie_map = food_calorie_map.drop_duplicates(subset=['name'])
food_calorie_map = food_calorie_map.sort_values('name')

# Save the calorie mapping
food_calorie_map.to_csv('food_calorie_map.csv', index=False)
print(f"Created calorie mapping for {len(food_calorie_map)} food items")
print(f"Average calories: {food_calorie_map['calories'].mean():.2f}")
print(f"Calorie range: {food_calorie_map['calories'].min()} - {food_calorie_map['calories'].max()}")

# Display sample
print("\nSample calorie mappings:")
print(food_calorie_map.head(10))
