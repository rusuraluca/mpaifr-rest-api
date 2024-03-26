from flask import Flask, request, jsonify
from utils.api_utils import process_images_and_compute_similarity


app = Flask(__name__)


@app.route('/')
def home():
    return "Face Recognition API"


@app.route('/images_similarity', methods=['POST'])
def get_similarity():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please provide two images."}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1 and file2:
        similarity = process_images_and_compute_similarity(file1, file2)
        return jsonify({"similarity": similarity})
    else:
        return jsonify({"error": "Invalid files provided."}), 400


if __name__ == '__main__':
    app.run(debug=True)
