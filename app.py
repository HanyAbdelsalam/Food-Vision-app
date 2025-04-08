from flask import Flask, render_template, url_for, request, jsonify
from functions import *


app = Flask(__name__, template_folder='templates', static_folder='static')

model = build_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    if "file" not in request.files:
        print("No file part in request.")
        return jsonify({"error": "No file part"}), 400  

    img = request.files["file"]

    if img.filename == "":
        print("No selected file.")
        return jsonify({"error": "No selected file"}), 400  

    print(f"📂 Received file: {img.filename}")
    
    delete_top5_image()
    
    try:
        processed_img = prepare_img(img)
        print(f"Prepared image shape: {processed_img.shape}")

        predictions = model.predict(processed_img)
        print("🔢 Raw model output:", predictions)
        
        predicted_class, confidence = get_prediction_label(predictions, CLASS_NAMES)
        plot_top_5(predictions, CLASS_NAMES)
        print(f"Predicion: {predicted_class}, Confidence: {confidence}")
        return jsonify({
                "class": predicted_class,
                "confidence": round(confidence, 2)
            })
        
    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": "Error processing image"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
