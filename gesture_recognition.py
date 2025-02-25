import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import os

class GestureRecognition:
    def __init__(self, num_channels=4, window_size=100):
        """
        Initialize the gesture recognition system
        num_channels: Number of sensor channels (one per finger)
        window_size: Number of samples to use for feature extraction
        """
        self.num_channels = num_channels
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100)
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def extract_features(self, signals):
        """
        Extract features from raw signals
        signals: List of signal arrays, one per channel
        """
        features = []
        for signal in signals:
            # Basic statistical features
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75)
            ])
            
            # Frequency domain features
            fft = np.abs(np.fft.fft(signal))
            features.extend([
                np.mean(fft),
                np.std(fft),
                np.max(fft)
            ])
        
        return np.array(features)
    
    def prepare_training_data(self, data_folder):
        """
        Prepare training data from recorded gestures
        data_folder: Folder containing gesture recording files
        """
        X = []
        y = []
        
        # Process each gesture recording
        for gesture_folder in os.listdir(data_folder):
            gesture_path = os.path.join(data_folder, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue
            
            # Load and process recordings for this gesture
            for recording in os.listdir(gesture_path):
                if not recording.endswith('.npz'):
                    continue
                    
                data = np.load(os.path.join(gesture_path, recording))
                signals = [data[f'channel_{i}'] for i in range(self.num_channels)]
                
                # Extract features from the recording
                features = self.extract_features(signals)
                X.append(features)
                y.append(gesture_folder)  # Use folder name as gesture label
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """
        Train the gesture recognition model
        X: Feature matrix
        y: Gesture labels
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        self.is_trained = True
        return test_score
    
    def save_model(self, model_name=None):
        """
        Save the trained model and scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        if model_name is None:
            model_name = f"gesture_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model and scaler
        model_path = os.path.join('models', f"{model_name}.joblib")
        scaler_path = os.path.join('models', f"{model_name}_scaler.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved as {model_path}")
    
    def load_model(self, model_name):
        """
        Load a trained model and scaler
        """
        model_path = os.path.join('models', f"{model_name}.joblib")
        scaler_path = os.path.join('models', f"{model_name}_scaler.joblib")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
    
    def predict_gesture(self, signals):
        """
        Predict gesture from current signal window
        signals: List of signal arrays from each channel
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features from the current window
        features = self.extract_features(signals)
        
        # Scale features and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)
        confidence = np.max(probabilities)
        
        return prediction[0], confidence