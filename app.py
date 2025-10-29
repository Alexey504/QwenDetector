from flask import Flask, render_template, request, jsonify
from detector.inference import generate_visual_answer
from detector.model_loader import get_model_and_processor
import io, base64
from PIL import Image

app = Flask(__name__)

model, processor = get_model_and_processor()

@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file = request.files["image"]
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Введите запрос"}), 400

        image = Image.open(file.stream).convert("RGB")

        annotated_image, description = generate_visual_answer(image, prompt)

        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "image": f"data:image/png;base64,{img_base64}",
            "text": description
        })
    except Exception as e:
        print("Ошибка при обработке:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Запуск приложения на http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)

