from flask import Flask, request, jsonify, render_template
from repo_dev.model import CarVsTruck # import class created in model.py
import os
import logging

app = Flask(__name__)
classifier = CarVsTruck('model/best.pt')   #intialize classfier with best.pt file

# Folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Existing API endpoint ----------
@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    image_url = request.args.get("image_url")
    if not image_url:
        return jsonify({"error": "No image_url provided"}), 400
        
    prediction = classifier.predict(image_url)
    return jsonify({"predicted_class": str(prediction)})

# ---------- New HTML web interface ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400
        
        # Save the uploaded image
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Get prediction
        predicted_class = classifier.predict(filepath)
        return f"<h2>Predicted Class: {predicted_class}</h2>"
    
    # Render the HTML form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001) #using run so it can run when deployment 