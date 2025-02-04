import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
import os
import warnings
import time
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import librosa.display
from joblib import load
import base64
from scipy.stats import zscore
warnings.filterwarnings("ignore")

class ParkinsonsVoiceAnalyzer:
    def __init__(self, model_path='gradient_boosting_model.joblib'):
        self.model = load(model_path)
        self.sample_rate = 44100
        self.duration = 5
        
        self.feature_ranges = {
            'MDVP:Fo(Hz)': (88.333, 260.105),
            'MDVP:Fhi(Hz)': (102.145, 592.030),
            'MDVP:Flo(Hz)': (65.476, 239.170),
            'MDVP:Jitter(%)': (0.00168, 0.03316),
            'MDVP:Jitter(Abs)': (0.000007, 0.00026),
            'MDVP:RAP': (0.00068, 0.02144),
            'MDVP:PPQ': (0.00092, 0.01958),
            'Jitter:DDP': (0.00204, 0.06433),
            'MDVP:Shimmer': (0.00954, 0.11908),
            'MDVP:Shimmer(dB)': (0.085, 1.302),
            'Shimmer:APQ3': (0.00455, 0.05647),
            'Shimmer:APQ5': (0.00576, 0.0794),
            'MDVP:APQ': (0.00719, 0.13778),
            'Shimmer:DDA': (0.01364, 0.16942),
            'NHR': (0.00065, 0.31482),
            'HNR': (8.441, 33.047),
            'RPDE': (0.25657, 0.685151),
            'DFA': (0.574282, 0.825288),
            'spread1': (-7.964984, -2.434031),
            'spread2': (0.006274, 0.450493),
            'D2': (1.423287, 3.671155),
            'PPE': (0.044539, 0.527367)
        }
    
    def record_voice(self):
        recording = sd.rec(int(self.duration * self.sample_rate), 
                         samplerate=self.sample_rate, 
                         channels=1)
        sd.wait()
        
        temp_path = 'temp_recording.wav'
        wav.write(temp_path, self.sample_rate, recording)
        return temp_path
    
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract pitch (f0) using librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                       fmin=librosa.note_to_hz('C2'), 
                                                       fmax=librosa.note_to_hz('C7'))
            f0 = f0[voiced_flag]
            
            if len(f0) == 0:
                raise ValueError("No valid pitch detected. Please speak louder or check your microphone.")
            
            # Calculate basic frequency features
            f0_mean = np.mean(f0)
            f0_max = np.max(f0)
            f0_min = np.min(f0)
            
            # Calculate jitter features
            period = 1.0 / f0
            jitter_local = np.mean(np.abs(np.diff(period))) / np.mean(period)
            jitter_abs = np.mean(np.abs(np.diff(period)))
            
            # Calculate shimmer features using amplitude envelope
            amplitude_env = np.abs(librosa.stft(y))
            shimmer_local = np.mean(np.abs(np.diff(amplitude_env, axis=1))) / np.mean(amplitude_env)
            shimmer_db = 20 * np.log10(shimmer_local + 1e-10)
            
            # Calculate harmonicity features
            harmonicity = librosa.feature.spectral_flatness(y=y)
            hnr = -10 * np.log10(np.mean(harmonicity) + 1e-10)
            nhr = 1 / (hnr + 1e-10)
            
            features = {
                'MDVP:Fo(Hz)': f0_mean,
                'MDVP:Fhi(Hz)': f0_max,
                'MDVP:Flo(Hz)': f0_min,
                'MDVP:Jitter(%)': jitter_local * 100,
                'MDVP:Jitter(Abs)': jitter_abs,
                'MDVP:RAP': jitter_local * 0.5,
                'MDVP:PPQ': jitter_local * 0.6,
                'Jitter:DDP': jitter_local * 1.5,
                'MDVP:Shimmer': shimmer_local,
                'MDVP:Shimmer(dB)': shimmer_db,
                'Shimmer:APQ3': shimmer_local * 0.5,
                'Shimmer:APQ5': shimmer_local * 0.7,
                'MDVP:APQ': shimmer_local * 0.9,
                'Shimmer:DDA': shimmer_local * 1.5,
                'NHR': nhr,
                'HNR': hnr,
                'RPDE': self.calculate_rpde(f0),
                'DFA': self.calculate_dfa(f0),
                'spread1': np.log(np.percentile(f0, 75) - np.percentile(f0, 25)),
                'spread2': np.std(f0),
                'D2': self.calculate_correlation_dimension(f0),
                'PPE': self.calculate_ppe(f0)
            }
            
            return pd.DataFrame([features])
            
        except Exception as e:
            raise ValueError(f"Error extracting features: {str(e)}")
    
    def scale_features(self, features_df):
        scaled_data = features_df.copy()
        
        for column in scaled_data.columns:
            min_val, max_val = self.feature_ranges[column]
            scaled_data[column] = -1 + 2 * (scaled_data[column] - min_val) / (max_val - min_val)
            scaled_data[column] = scaled_data[column].clip(-1, 1)
            
        return scaled_data
    
    def calculate_rpde(self, pitch_values):
        try:
            hist, _ = np.histogram(pitch_values, bins=50, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        except:
            return 0.5
    
    def calculate_dfa(self, pitch_values):
        try:
            cumsum = np.cumsum(pitch_values - np.mean(pitch_values))
            return np.std(cumsum)
        except:
            return 0.7
    
    def calculate_correlation_dimension(self, pitch_values):
        try:
            return np.log(len(np.unique(pitch_values))) / np.log(len(pitch_values))
        except:
            return 2.0
    
    def calculate_ppe(self, pitch_values):
        try:
            hist, _ = np.histogram(np.diff(pitch_values), bins=50, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        except:
            return 0.2
    
    def predict(self, features):
        try:
            scaled_features = self.scale_features(features)
            prediction = self.model.predict(scaled_features)
            probability = self.model.predict_proba(scaled_features)
            max_prob = max(probability[0]) * 100
            
            final_prediction = 1 if (prediction[0] == 1 and max_prob >= 98.0) else 0
            return final_prediction, probability[0]
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
# Database
class DatabaseManager:
    def __init__(self, db_name='voice_analysis.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()
    
    def create_tables(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                notes TEXT,
                audio_data BLOB
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER,
                feature_name TEXT,
                feature_value REAL,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
            )
        ''')
        self.conn.commit()

    def save_recording(self, audio_data, prediction, confidence, features, notes=""):
        c = self.conn.cursor()
        
        # Save recording
        c.execute('''
            INSERT INTO recordings (timestamp, prediction, confidence, notes, audio_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction,
            confidence,
            notes,
            audio_data
        ))
        
        recording_id = c.lastrowid
        
        # Save features
        for name, value in features.items():
            c.execute('''
                INSERT INTO features (recording_id, feature_name, feature_value)
                VALUES (?, ?, ?)
            ''', (recording_id, name, float(value)))
        
        self.conn.commit()
        return recording_id

    def get_recording_features(self, recording_id):
        c = self.conn.cursor()
        features = c.execute('''
            SELECT feature_name, feature_value 
            FROM features 
            WHERE recording_id = ?
        ''', (recording_id,)).fetchall()
        return {name: value for name, value in features}

    def get_all_recordings(self):
        c = self.conn.cursor()
        recordings = c.execute('''
            SELECT id, timestamp, prediction, confidence, notes 
            FROM recordings 
            ORDER BY timestamp DESC
        ''').fetchall()
        
        result = []
        for rec in recordings:
            features = self.get_recording_features(rec[0])
            result.append({
                'id': rec[0],
                'timestamp': rec[1],
                'prediction': rec[2],
                'confidence': rec[3],
                'notes': rec[4],
                'features': features
            })
        return result

    def delete_recording(self, recording_id):
        c = self.conn.cursor()
        c.execute('DELETE FROM features WHERE recording_id = ?', (recording_id,))
        c.execute('DELETE FROM recordings WHERE id = ?', (recording_id,))
        self.conn.commit()

    def clear_all_history(self):
        c = self.conn.cursor()
        c.execute('DELETE FROM features')
        c.execute('DELETE FROM recordings')
        self.conn.commit()

def display_history(db):
    st.sidebar.markdown("<h2 style='color: #FF0000;'>Analysis History</h2>", unsafe_allow_html=True)
    
    if st.sidebar.button("Clear All History", key="clear_all"):
        if st.sidebar.checkbox("Confirm deletion of all records?", key="confirm_clear"):
            db.clear_all_history()
            st.sidebar.success("All history cleared!")
            st.experimental_rerun()
    
    recordings = db.get_all_recordings()
    
    for record in recordings:
        with st.sidebar.expander(f"Recording {record['timestamp']}"):
            st.write(f"Prediction: {record['prediction']}")
            st.write(f"Confidence: {record['confidence']:.2f}%")
            
            key_metrics = {
                'Fundamental Frequency': record['features'].get('MDVP:Fo(Hz)', 0),
                'Jitter': record['features'].get('MDVP:Jitter(%)', 0),
                'Shimmer': record['features'].get('MDVP:Shimmer', 0),
                'HNR': record['features'].get('HNR', 0)
            }
            
            st.markdown("### Key Metrics")
            for metric, value in key_metrics.items():
                st.write(f"{metric}: {value:.4f}")
            
            # Notes section
            notes = record['notes'] if record['notes'] else ""
            new_notes = st.text_area("Notes", value=notes, key=f"notes_{record['id']}")
            
            col1, col2 = st.columns(2)
            
            # Save notes button
            if col1.button("Save Notes", key=f"save_{record['id']}"):
                c = db.conn.cursor()
                c.execute('UPDATE recordings SET notes = ? WHERE id = ?', 
                         (new_notes, record['id']))
                db.conn.commit()
                st.success("Notes saved!")
            
            # Delete recording button
            if col2.button("Delete", key=f"delete_{record['id']}"):
                db.delete_recording(record['id'])
                st.success("Recording deleted!")
                st.experimental_rerun()

# Visualization Functions
class Visualizer:
    @staticmethod
    def create_waveform_plot(audio_data, sample_rate):
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Waveform', 'Spectrogram'))
        
        # Waveform
        time_points = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        fig.add_trace(
            go.Scatter(x=time_points, y=audio_data.flatten(), 
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data.flatten())))
        fig.add_trace(
            go.Heatmap(z=D, colorscale='Viridis'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @staticmethod
    def create_feature_radar_plot(features):
        selected_features = {
            'Fundamental Frequency': features['MDVP:Fo(Hz)'],
            'Jitter': features['MDVP:Jitter(%)'],
            'Shimmer': features['MDVP:Shimmer'],
            'HNR': features['HNR'],
            'DFA': features['DFA'],
            'PPE': features['PPE']
        }
        
        values = list(selected_features.values())
        normalized_values = zscore(values)
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=list(selected_features.keys()),
            fill='toself',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min(normalized_values)-1, max(normalized_values)+1]
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @staticmethod
    def create_metrics_dashboard(features):
        col1, col2, col3 = st.columns(3)
        
        metrics = {
            'Voice Stability': {
                'Jitter (%)': features['MDVP:Jitter(%)'],
                'Shimmer': features['MDVP:Shimmer'],
                'HNR': features['HNR']
            },
            'Frequency Analysis': {
                'F0 (Hz)': features['MDVP:Fo(Hz)'],
                'Fhi (Hz)': features['MDVP:Fhi(Hz)'],
                'Flo (Hz)': features['MDVP:Flo(Hz)']
            },
            'Advanced Metrics': {
                'RPDE': features['RPDE'],
                'DFA': features['DFA'],
                'D2': features['D2']
            }
        }
        
        columns = [col1, col2, col3]
        for idx, (category, values) in enumerate(metrics.items()):
            with columns[idx]:
                st.markdown(f"<h4 style='color: #FF0000;'>{category}</h4>", 
                          unsafe_allow_html=True)
                for name, value in values.items():
                    st.markdown(
                        f"""
                        <div style='background-color: #2D2D2D; 
                                  padding: 10px; 
                                  border-radius: 5px; 
                                  margin: 5px 0;
                                  animation: fadeIn 0.5s;'>
                            <span style='color: #FF0000;'>{name}:</span>
                            <br>
                            <span style='font-size: 1.2em;'>{value:.4f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# CSS
def apply_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #8B0000;
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FF0000;
            transform: scale(1.05);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .recording-animation {
            animation: pulse 1s infinite;
        }
        .metric-card {
            animation: fadeIn 0.5s;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Parkinson's Voice Analyzer", 
                      layout="wide",
                      initial_sidebar_state="expanded")
    
    apply_custom_style()
    db = DatabaseManager()
    
    display_history(db)
    
    # Main content
    st.markdown(
        "<h1 style='text-align: center; color: #FF0000; animation: fadeIn 1s;'>"
        "Parkinson's Voice Analyzer"
        "</h1>", 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div style='background-color: #2D2D2D; 
                      padding: 20px; 
                      border-radius: 10px;
                      animation: fadeIn 1s;'>
                <h3 style='color: #FF0000;'>Recording Instructions</h3>
                <ul>
                    <li>Ensure you're in a quiet environment</li>
                    <li>Position your mouth ~6 inches from the microphone</li>
                    <li>Say 'aaah' with a steady tone</li>
                    <li>Maintain comfortable volume</li>
                    <li>Recording duration: 5 seconds</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        analyzer = ParkinsonsVoiceAnalyzer('gradient_boosting_model.joblib')
        
        if st.button("Start Recording", key="record_button"):
            with st.spinner("Preparing to record..."):
                # Countdown animation
                for i in range(3, 0, -1):
                    st.markdown(
                        f"<h1 style='text-align: center; color: #FF0000; "
                        f"animation: pulse 1s infinite;'>{i}</h1>",
                        unsafe_allow_html=True
                    )
                    time.sleep(1)
                
                st.markdown(
                    "<div class='recording-animation'>"
                    "<h2 style='text-align: center; color: #FF0000;'>"
                    "Recording..."
                    "</h2></div>",
                    unsafe_allow_html=True
                )
                
                # Record and analyze
                try:
                    audio_path = analyzer.record_voice()
                    features = analyzer.extract_features(audio_path)
                    prediction, probabilities = analyzer.predict(features)
                    confidence = max(probabilities) * 100
                    
                    # Save to database
                    with open(audio_path, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    recording_id = db.save_recording(
                        audio_data,
                        'Parkinson\'s Detected' if prediction == 1 else 'No Parkinson\'s Detected',
                        confidence,
                        features.iloc[0].to_dict()
                    )
                    
                    # Display results
                    result_color = '#FF0000' if prediction == 1 else '#00FF00'
                    st.markdown(
                        f"<h2 style='text-align: center; color: {result_color};'>"
                        f"{'Parkinson\'s Detected' if prediction == 1 else 'No Parkinson\'s Detected'}"
                        f"</h2>",
                        unsafe_allow_html=True
                    )
                    
                    # Load and display audio visualization
                    y, sr = librosa.load(audio_path)
                    st.plotly_chart(
                        Visualizer.create_waveform_plot(y, sr),
                        use_container_width=True
                    )
                    
                    # Clean up
                    os.remove(audio_path)
                    
                except Exception as e:
                    st.error(f"Error during recording: {str(e)}")
    
    # Feature Analysis Column
    with col2:
        if 'features' in locals():
            st.markdown("<h3 style='color: #FF0000;'>Feature Analysis</h3>", 
                       unsafe_allow_html=True)
            
            # Radar plot
            st.plotly_chart(
                Visualizer.create_feature_radar_plot(features.iloc[0]),
                use_container_width=True
            )
            
            # Metrics dashboard
            Visualizer.create_metrics_dashboard(features.iloc[0])

if __name__ == "__main__":
    main()