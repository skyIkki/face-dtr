from flask import Flask, request, send_from_directory, render_template_string, send_file
import os
import io
import zipfile
from datetime import datetime

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    label = request.form.get('label')
    image = request.files.get('image')

    if not label or not image:
        return "Missing label or image", 400

    save_dir = os.path.join("training_data", label)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"{timestamp}.jpg")
    image.save(path)

    return "Uploaded successfully", 200

@app.route('/list')
def list_images():
    root_dir = "training_data"
    image_data = {}

    if not os.path.exists(root_dir):
        return "<h2>❗No uploaded images yet.</h2>"

    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            images = os.listdir(label_dir)
            image_data[label] = images

    html = """
    <h2>🌼 Uploaded Flower Images</h2>
    <form method="POST" action="/clear">
        <button type="submit" style="margin-bottom: 20px;">🗑️ Clear All Images</button>
    </form>
    {% for label, images in image_data.items() %}
        <h3>{{ label }}</h3>
        {% for img in images %}
            <div style="display:inline-block;margin:10px;">
                <img src="/image/{{ label }}/{{ img }}" height="150"><br>
                {{ img }}
            </div>
        {% endfor %}
    {% endfor %}
    """
    return render_template_string(html, image_data=image_data)

@app.route('/image/<label>/<filename>')
def serve_image(label, filename):
    return send_from_directory(os.path.join("training_data", label), filename)

@app.route('/download-data')
def download_data():
    data_dir = 'training_data'
    if not os.path.exists(data_dir):
        return "No training data available.", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                zf.write(full_path, arcname=rel_path)
    memory_file.seek(0)
    return send_file(memory_file, download_name="training_data.zip", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
