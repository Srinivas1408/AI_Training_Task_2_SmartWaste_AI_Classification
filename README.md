# SmartWaste AI — Task 2

An image-classification API that predicts which of six waste categories an uploaded image belongs to. The model uses transfer learning with MobileNetV2 and is served through FastAPI.

## 1. Project objective

Build an API that accepts a waste image and returns the predicted category, a confidence score, and scores for all six categories.

**Supported classes:** `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

## 2. Model and dataset

- **Model:** ImageNet-pretrained MobileNetV2 used as a feature extractor, with a custom classification head.
- **Input:** RGB image resized to `224 × 224` pixels.
- **Output:** scores for six classes.
- **Preprocessing:** `tf.keras.applications.mobilenet_v2.preprocess_input`.
- **Dataset:** Garbage Classification image dataset.
- **Split:** stratified training, validation, and test sets (70% / 15% / 15%).

Use only one copy of the dataset. Duplicate copies can cause data leakage if similar images appear in different splits. The saved model is named `smartwaste_model.keras`.

## 3. Packages and their purpose

| Package | Purpose |
|---|---|
| `tensorflow` | Load/train the neural network, preprocess images, and run predictions |
| `numpy` | Image arrays and prediction scores |
| `Pillow` | Open, convert, and resize uploaded images |
| `fastapi` | Define the API endpoints |
| `uvicorn[standard]` | Run the FastAPI server |
| `python-multipart` | Handle image uploads in POST requests |
| `scikit-learn` | Stratified splitting and evaluation metrics |
| `matplotlib` | Plot training curves and evaluation charts |
| `seaborn` | Optional confusion-matrix visualization |
| `kaggle` | Optional download of the dataset from Kaggle |
| `requests` | Test API endpoints from Python |

## 4. Suggested project structure

```text
SmartWasteAI/
├── app.py
├── smartwaste_model.keras
├── requirements.txt
└── README.md
```

The model file may be large, so it can be stored separately from source control. In the Colab workflow, the API currently expects `/content/smartwaste_model.keras`; update `MODEL_PATH` in `app.py` if you use a different location.

## 5. Install dependencies

Use a Python version supported by your selected TensorFlow release. Python 3.10 or 3.11 is a common starting point for local setup; verify compatibility for your OS and hardware. Google Colab often already includes TensorFlow.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For API-only inference, the core packages are `tensorflow`, `numpy`, `Pillow`, `fastapi`, `uvicorn[standard]`, and `python-multipart`. Other packages support training, evaluation, dataset download, or testing.

## 6. Run the API

From the directory containing `app.py` and the model, run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

While the server is running, these endpoints are available:

- `http://127.0.0.1:8000/` — home endpoint
- `http://127.0.0.1:8000/health` — health check
- `http://127.0.0.1:8000/docs` — interactive Swagger UI

In Google Colab, create a browser-accessible proxy URL:

```python
from google.colab.output import eval_js
proxy_url = eval_js("google.colab.kernel.proxyPort(8000)")
print(proxy_url + "/docs")
```

## 7. Test an image in Swagger UI

1. Open the printed `/docs` URL.
2. Expand **POST /predict**.
3. Click **Try it out**.
4. Click **Choose File** and select a JPG, JPEG, or PNG image.
5. Click **Execute**.
6. Review the response body and status code.

Example response (illustrative values only):

```json
{
  "predicted_category": "plastic",
  "confidence": 96.42,
  "probabilities": {
    "cardboard": 0.01,
    "glass": 0.01,
    "metal": 0.01,
    "paper": 0.01,
    "plastic": 0.9642,
    "trash": 0.0088
  }
}
```

## 8. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic API message and supported categories |
| `GET` | `/health` | Check whether the service and model are loaded |
| `POST` | `/predict` | Upload an image and receive a classification |

## 9. Common issues

- **Connection refused:** confirm Uvicorn is still running and wait for startup; inspect the server log if it fails.
- **HTTP 422:** verify that an image was selected and the upload field matches the parameter in `app.py`.
- **HTTP 500:** inspect the server log; the model path, image decoding, or preprocessing may be the cause.
- **Model not found:** update `MODEL_PATH` to the actual model location.
- **Unsupported image:** upload a valid JPG, JPEG, or PNG file.
- **Inconsistent predictions:** use clear images with good lighting and one main object. Confidence scores are not a guarantee of correctness.

## 10. Security and deployment notes

- Never publish Kaggle API tokens or other secrets in notebooks or source control. Use Colab Secrets or environment variables.
- The development server is intended for testing. Public deployment should include request-size limits, robust image validation, and suitable access controls.
- A confidence score is the model's score, not necessarily a calibrated probability.

## 11. Requirements summary

The accompanying `requirements.txt` lists packages for API serving, model inference, training/evaluation, and optional Kaggle dataset access. For reproducible deployment, test compatible package versions in the target environment and pin versions after successful testing.
