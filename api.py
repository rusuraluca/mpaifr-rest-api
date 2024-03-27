from flask import Flask, request, jsonify
from utils.api_utils import process_images_and_compute_similarity


app = Flask(__name__)


@app.route('/')
def home():
    return "Face Recognition API"


@app.route('/images_similarity', methods=['POST'])
def get_similarity():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"invalid_request_error": "Please provide two images."}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if not file1 or not file2:
        return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    if file1 and file2:
        similarity = process_images_and_compute_similarity(file1, file2)
        return jsonify({"similarity": round(similarity, 2)})
    else:
        return jsonify({"request_error": "The parameters were valid but the request failed."}), 402


@app.route('/batch_images_similarity', methods=['POST'])
def get_batch_similarity():
    if 'image' not in request.files:
        return jsonify({"invalid_request_error": "Please provide the main image."}), 400

    image_list_files = request.files.getlist('imageList')

    if not image_list_files:
        return jsonify({"invalid_request_error": "Please provide a list of images."}), 400

    file1 = request.files['image']
    similarities = []

    for file2 in image_list_files:
        if file1 and file2:
            similarity = process_images_and_compute_similarity(file1, file2)
            similarities.append(similarity)
        else:
            return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    if similarities:
        average_similarity = sum(similarities) / len(similarities)
        return jsonify({"similarity": round(average_similarity, 2)})
    else:
        return jsonify({"request_error": "The parameters were valid but the request failed."}), 402


if __name__ == '__main__':
    app.run(debug=True)
