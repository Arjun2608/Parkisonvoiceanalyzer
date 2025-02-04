Parkinson disease detection using voice

Dataset used: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set

Model Trained and saved : Gradient Boosting 

Backend and frontend : Streamlit (Python)
Database : SQLite

In terminal paste 
pip install requests pandas imbalanced-learn seaborn matplotlib numpy scikit-learn xgboost ipython streamlit sounddevice scipy plotly librosa joblib
and install all required libraies and modules

After installing type
streamlit run gui.py

A tab will open in the browser and u can start recording your voice . Speak "aahh" for 5 seconds loudly with clear voice it will calculate all required values and detect 0 or 1.

https://youtu.be/gNIdxYjGVV8?feature=shared use this video for verification(0.28 to 0.39 for Parkinson detected and 1.11to 1.21 for not detected) . 
