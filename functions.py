import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensure Agg backend is used before importing pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


def build_model():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
    
    backbone = EfficientNetB0(include_top=False)
    for layer in backbone.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            layer.kernel_regularizer = tf.keras.regularizers.l2(0.001)
    
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_layer')(x)
    x = tf.keras.layers.Dense(101, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='logits')(x)
    outputs = tf.keras.layers.Activation('softmax', name='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="Food_Vision_Big")

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])

    model.load_weights("./Food_Vision_Final_epoch05.weights.h5")
    print("✅ Model Loaded Successfully")
    
    if not os.path.exists("./Food_Vision_Final_epoch05.weights.h5"):
        print("❌ Model weights file not found!")

    return model


def prepare_img(img):
    img_bytes = img.read()
    image = tf.image.decode_image(img_bytes, channels=3)
    print(f"Decoded image shape: {image.shape}")  # Log the shape after decoding
    image = tf.image.resize(image, (224, 224))
    print(f"Resized image shape: {image.shape}")  # Log the shape after resizing
    print(f"Image min value: {tf.reduce_min(image)}, max value: {tf.reduce_max(image)}")  # Log min/max values after normalization
    image = tf.expand_dims(image, axis=0)
    print(f"Final image shape: {image.shape}")  # Log the final shape after expanding dims
    return image


def get_prediction_label(prediction, class_names):
    """Returns the class label and confidence score for the highest predicted class."""
    predicted_index = np.argmax(prediction)  # Get index of highest probability
    predicted_label = class_names[predicted_index]  # Get corresponding class name
    confidence = prediction[0][predicted_index] * 100  # Convert to percentage
    return predicted_label, confidence


def plot_top_5(prediction, class_names):
    prediction = prediction.squeeze()
    top5_indices = np.argsort(prediction)[-5:][::-1]
    top5_probs = prediction[top5_indices]
    top5_classes = [class_names[i] for i in top5_indices]

    plt.figure(figsize=(12, 8), dpi=100, facecolor='none')  # Larger size for better web visibility
    plt.barh(top5_classes[::-1], top5_probs[::-1], color='skyblue')

    plt.xlabel("Probability", color='white')
    plt.title("Top 5 Predicted Classes", color='white')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7, color='white')

    plt.yticks(rotation=15, fontsize=12, color='white')
    plt.xticks(fontsize=12, color='white')

    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.2)

    plt.savefig('static/images/top_5.png', bbox_inches='tight', pad_inches=0.3, transparent=True, dpi=100)
    plt.close()


def delete_top5_image():
    image_path = "static/images/top_5.png"
    
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        else:
            print(f"File does not exist: {image_path}")
    except PermissionError:
        print(f"Permission denied: Unable to delete {image_path}. Check file permissions.")
    except Exception as e:
        print(f"An error occurred while deleting {image_path}: {e}")


# Global class names list
CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
    "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros",
    "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
    "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette",
    "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
    "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
    "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]