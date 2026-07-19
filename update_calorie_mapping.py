import pandas as pd

# Load existing calorie mapping
calorie_df = pd.read_csv('food_calorie_map.csv')

# Define the 20 trained classes with their calorie estimates
trained_classes = {
    'Aloo fry': 250,
    'Beetroot': 43,
    'Biryani': 350,
    'Bitter gourd': 90,
    'Butter chicken': 450,
    'Dal - Curry': 180,
    'Dosa': 170,
    'Gulab Jamun': 350,
    'Idly': 60,
    'Jalebi': 400,
    'Kebab': 280,
    'Mango Pickle': 150,
    'Palak Paneer': 320,
    'Puri': 150,
    'Rajma chawal': 280,
    'Rasmalai': 380,
    'Roti': 100,
    'Sambar': 140,
    'Samosa': 260,
    'Vada': 140
}

# Create a new DataFrame for trained classes with correct course classifications
trained_calories = []
for food_name, calories in trained_classes.items():
    # Determine diet type
    diet = 'vegetarian' if food_name not in ['Butter chicken', 'Kebab'] else 'non vegetarian'
    
    # Determine course type more accurately
    main_course = ['Biryani', 'Butter chicken', 'Dal - Curry', 'Palak Paneer', 'Rajma chawal', 'Sambar', 'Aloo fry', 'Bitter gourd']
    snack = ['Samosa', 'Vada', 'Puri', 'Idly', 'Dosa', 'Roti', 'Beetroot', 'Mango Pickle']
    dessert = ['Gulab Jamun', 'Jalebi', 'Rasmalai', 'Kebab']
    
    if food_name in main_course:
        course = 'main course'
    elif food_name in snack:
        course = 'snack'
    elif food_name in dessert:
        course = 'dessert'
    else:
        course = 'main course'  # default
    
    trained_calories.append({
        'name': food_name,
        'calories': calories,
        'diet': diet,
        'course': course
    })

trained_df = pd.DataFrame(trained_calories)

# Merge with existing calorie data, prioritizing trained classes
# Remove any existing entries that match trained classes
calorie_df = calorie_df[~calorie_df['name'].isin(trained_classes.keys())]

# Combine
combined_df = pd.concat([trained_df, calorie_df], ignore_index=True)

# Save updated mapping
combined_df.to_csv('food_calorie_map.csv', index=False)

print(f"Updated calorie mapping with {len(trained_classes)} trained classes")
print(f"Total food items in database: {len(combined_df)}")
print("\nTrained classes with calories:")
for food_name, calories in trained_classes.items():
    print(f"{food_name}: {calories} kcal")
