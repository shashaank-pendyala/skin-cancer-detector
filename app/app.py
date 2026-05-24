import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
IMG_SIZE = 224
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

CLASS_LABELS = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

CLASS_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'risk': 'High',
        'description': 'Precancerous lesion caused by sun damage. Requires medical attention.',
        'color': 'danger'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'risk': 'High',
        'description': 'Most common type of skin cancer. Rarely spreads but needs treatment.',
        'color': 'danger'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'risk': 'Low',
        'description': 'Non-cancerous skin growth. Generally harmless but monitor for changes.',
        'color': 'success'
    },
    'df': {
        'name': 'Dermatofibroma',
        'risk': 'Low',
        'description': 'Benign skin nodule. Usually harmless and requires no treatment.',
        'color': 'success'
    },
    'mel': {
        'name': 'Melanoma',
        'risk': 'Very High',
        'description': 'Most dangerous form of skin cancer. Immediate medical consultation required.',
        'color': 'danger'
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'risk': 'Low',
        'description': 'Common mole. Usually benign but monitor for changes in size or color.',
        'color': 'success'
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'risk': 'Low',
        'description': 'Benign vascular growth. Generally harmless.',
        'color': 'success'
    }
}


def build_model():
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Lambda(
        lambda img: preprocess_input(img),
        name='preprocessing'
    )(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(7, activation='softmax')(x)
    return Model(inputs, outputs)


def load_model_weights():
    model = build_model()
    weights_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'model',
        'efficientnet_ham10000_weights.weights.h5'
    )
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Model weights loaded successfully")
    else:
        print(f"WARNING: Weights not found at {weights_path}")
    return model


def make_gradcam_heatmap(img_array, model, pred_index=None):
    efficientnet = model.get_layer('efficientnetb0')
    last_conv_layer = efficientnet.get_layer('top_activation')
    efficientnet_grad_model = tf.keras.models.Model(
        inputs=efficientnet.inputs,
        outputs=[last_conv_layer.output, efficientnet.output]
    )
    preprocess_layer = model.get_layer('preprocessing')

    with tf.GradientTape() as tape:
        preprocessed = preprocess_layer(img_array)
        conv_outputs, features = efficientnet_grad_model(preprocessed)
        tape.watch(conv_outputs)
        x = model.get_layer('global_average_pooling2d')(features)
        x = model.get_layer('dropout')(x, training=False)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout_1')(x, training=False)
        preds = model.get_layer('dense_1')(x)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def generate_gradcam_image(img_path, heatmap, save_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    superimposed_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, superimposed_bgr)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load model at startup
print("Loading model...")
model = load_model_weights()
print("Model ready")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400

    # Save uploaded image
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{file.filename}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # Preprocess
    img = tf.keras.preprocessing.image.load_img(
        upload_path, target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array_expanded, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    predicted_class = CLASS_LABELS[predicted_idx]

    # Top 3 predictions
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3 = [
        {
            'class': CLASS_LABELS[i],
            'name': CLASS_INFO[CLASS_LABELS[i]]['name'],
            'confidence': round(float(predictions[0][i]) * 100, 1)
        }
        for i in top3_idx
    ]

    # Grad-CAM
    gradcam_filename = f"gradcam_{unique_id}.jpg"
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    try:
        heatmap, _ = make_gradcam_heatmap(img_array_expanded, model, predicted_idx)
        generate_gradcam_image(upload_path, heatmap, gradcam_path)
        gradcam_url = f"static/uploads/{gradcam_filename}"
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gradcam_url = None

    result = {
        'predicted_class': predicted_class,
        'class_name': CLASS_INFO[predicted_class]['name'],
        'risk': CLASS_INFO[predicted_class]['risk'],
        'description': CLASS_INFO[predicted_class]['description'],
        'color': CLASS_INFO[predicted_class]['color'],
        'confidence': round(confidence, 1),
        'top3': top3,
        'image_url': f"static/uploads/{filename}",
        'gradcam_url': gradcam_url
    }

    return render_template('result.html', result=result)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)