import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from tensorflow.keras.models import load_model
from scipy.stats import mode
from typing import List, Dict

# *** Import HTMLResponse ***
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Epileptic Seizure Ensemble API",
    description="Runs 4 models (XGB, RNN, LSTM, CNN-LSTM) for seizure detection."
)
try:
    SCALER = joblib.load('binary_scaler.gz')
    MODEL_XGB = joblib.load('xgboost_binary.pkl')
    MODEL_RNN = load_model('rnn_binary.keras')
    MODEL_LSTM = load_model('lstm_binary.keras')
    MODEL_CNN_LSTM = load_model('cnn_lstm_binary.keras')
    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please make sure all 5 model/scaler files are in the same folder as main.py")
    exit()

CLASS_NAMES = {
    0: 'Class 0 (Non-Seizure)',
    1: 'Class 1 (Seizure)'
}

class SignalInput(BaseModel):
    data: conlist(float, min_length=178, max_length=178)

class ModelPredictions(BaseModel):
    xgboost: str
    rnn: str
    lstm: str
    cnn_lstm: str

class EnsembleResponse(BaseModel):
    model_predictions: ModelPredictions
    ensemble_vote: str
    final_class_id: int

def preprocess_input(data_1d: np.ndarray):
    data_scaled_2d = SCALER.transform(data_1d.reshape(1, -1))
    data_scaled_3d = data_scaled_2d.reshape(1, 178, 1)
    return data_scaled_2d, data_scaled_3d
@app.post("/predict", response_model=EnsembleResponse)
async def predict(signal: SignalInput):
    try:
        raw_values = np.array(signal.data)
        processed_data_2d, processed_data_3d = preprocess_input(raw_values)
        pred_xgb = MODEL_XGB.predict(processed_data_2d)[0]
        pred_rnn = np.argmax(MODEL_RNN.predict(processed_data_3d), axis=1)[0]
        pred_lstm = np.argmax(MODEL_LSTM.predict(processed_data_3d), axis=1)[0]
        pred_cnn_lstm = np.argmax(MODEL_CNN_LSTM.predict(processed_data_3d), axis=1)[0]

        predictions = [int(pred_xgb), int(pred_rnn), int(pred_lstm), int(pred_cnn_lstm)]
        final_vote = mode(predictions, axis=0)[0]
        
        response = {
            'model_predictions': {
                'xgboost': CLASS_NAMES[pred_xgb],
                'rnn': CLASS_NAMES[pred_rnn],
                'lstm': CLASS_NAMES[pred_lstm],
                'cnn_lstm': CLASS_NAMES[pred_cnn_lstm]
            },
            'ensemble_vote': CLASS_NAMES[final_vote],
            'final_class_id': int(final_vote)
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EEG Seizure Detection</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: #f4f7f6; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
            #container { background: #fff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 2rem; width: 600px; }
            h1 { text-align: center; color: #333; }
            textarea { width: 95%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; margin-top: 1rem; }
            button { background: #007bff; color: white; border: none; padding: 12px 20px; border-radius: 4px; font-size: 16px; cursor: pointer; margin-top: 1rem; transition: background 0.3s ease; }
            button:hover { background: #0056b3; }
            #result { margin-top: 1.5rem; padding: 1rem; border: 1px solid #eee; border-radius: 4px; background: #fafafa; display: none; }
            #result pre { background: #eee; padding: 10px; border-radius: 4px; }
            .final-result { font-size: 1.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem; }
            .error { color: #D8000C; background: #FFD2D2; padding: 10px; border-radius: 4px; display: none; margin-top: 1rem; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>EEG Seizure Ensemble Analyzer</h1>
            <p>Type comma-separated EEG values into the box below.</p>
            <textarea id="eegData" placeholder="e.g., 135, 190, 229, 223, ..."></textarea>
            <button id="analyzeBtn">Analyze Signal</button>
            
            <div id="errorBox" class="error"></div>
            <div id="result">
                <div id="finalPrediction" class="final-result"></div>
                <h3>Individual Model Votes:</h3>
                <pre id="modelPredictions"></pre>
            </div>
        </div>

        <script>
            document.getElementById('analyzeBtn').addEventListener('click', async () => {
                const dataText = document.getElementById('eegData').value;
                const values = dataText.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
                
                const resultDiv = document.getElementById('result');
                const errorBox = document.getElementById('errorBox');
                const finalPredictionDiv = document.getElementById('finalPrediction');
                const modelPredsPre = document.getElementById('modelPredictions');

                // Reset UI
                resultDiv.style.display = 'none';
                errorBox.style.display = 'none';
                
                if (values.length !== 178) {
                    errorBox.textContent = `Error: Expected 178 numbers, but found ${values.length}.`;
                    errorBox.style.display = 'block';
                    return;
                }
                
                try {
                    // This fetch call points to the API on the *same server*
                    const apiUrl = '/predict';

                    const response = await fetch(apiUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ data: values })
                    });

                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.detail || `HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();

                    // Populate and show the results
                    finalPredictionDiv.textContent = `Final Result: ${result.ensemble_vote}`;
                    modelPredsPre.textContent = JSON.stringify(result.model_predictions, null, 2);
                    resultDiv.style.display = 'block';

                } catch (e) {
                    errorBox.textContent = `An error occurred: ${e.message}`;
                    errorBox.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)