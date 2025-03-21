import requests
from pandas import json_normalize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import pickle
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
import json

model_pickle_file = "isolation_forest_model.pkl"

def fetch_data(url):
    print("Fetching data")
    try:
        response = requests.post(url)
        response.raise_for_status()
        print("Logs retrieved!")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def anomaly_detection(data, model=None):
    
    print("saving in a file")
    with open('retrieved_logs.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("Data saved to 'data_retrieved.json'")
    print("Converting to CSV without saving into a file")

    # Convert JSON data to DataFrame
    df = json_normalize(
        data['logs'],
        record_path=None,
        meta=['timestamp', 'event_type'],
        errors='ignore'
    )

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z')

    # Calculate packets flow
    df['pps'] = df['flow.pkts_toserver'] + df['flow.pkts_toclient']
    df['bps'] = df['flow.bytes_toserver'] + df['flow.bytes_toclient']

    df.set_index('timestamp', inplace=True)

    # Resample data
    df_resampled = df.resample('1min').agg({
        'flow.pkts_toserver': 'sum',
        'flow.bytes_toserver': 'sum',
        'flow.pkts_toclient': 'sum',
        'flow.bytes_toclient': 'sum',
        'src_ip': 'first',
        'pps': 'mean',
        'bps': 'mean',
        'tcp.syn': 'sum',
        'tcp.ack': 'sum',
        'tcp.rst': 'sum',
        'flow.state': lambda x: x.mode()[0] if not x.mode().empty else None,
        'flow.reason': lambda x: x.mode()[0] if not x.mode().empty else None,
        'http.http_method': lambda x: x.mode()[0] if not x.mode().empty else None,
        'alert.signature_id': 'nunique',
        'alert.signature': 'nunique',
        'alert.category': 'nunique'
    })

    # Label Encoding for categorical columns
    label_encoder = LabelEncoder()
    categorical_columns = ['flow.state', 'flow.reason', 'http.http_method']

    for col in categorical_columns:
        if df_resampled[col].dtype == 'object':
            df_resampled[col] = label_encoder.fit_transform(df_resampled[col].astype(str))

    # Features for anomaly detection
    features = df_resampled[['pps', 'bps', 'tcp.syn', 'tcp.ack', 'tcp.rst', 
                            'flow.state', 'flow.reason', 'http.http_method', 
                            'alert.signature_id', 'alert.signature', 'alert.category']]

    # If no model is passed, create and train a new model
    if model is None:
        model = IsolationForest(contamination=0.01)
        model.fit(features)

    # Perform anomaly detection
    df_resampled['anomaly'] = model.predict(features)
    anomalies = df_resampled[df_resampled['anomaly'] == -1]

    # Save anomalies to CSV
    anomalies.to_csv("ddos4.csv", index=False)
    print("DDOS detected & stored in ddos4.csv")

    return model, anomalies  # Return model and anomalies for future use

def send_email():
    
    subject = "Potential DDOS Attack"
    sender_email = "sender@gmail.com"
    password = "password"
    receiver_email = "receiver@gmail.com"
    body = "Anomalies detected, pls find the attachment for anomaly logs"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    file_name = 'ddos4.csv'

    with open(file_name, "rb") as application:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(application.read())
    
    encoders.encode_base64(part)

    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {file_name}",
    )

    message.attach(part)
    text = message.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    print("Email sent!")


def save_model(model, filename):
    """Save the trained model to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load the trained model from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"No existing model found. A new model will be trained.")
        return None


def main():
    url = ''
    model = load_model(model_pickle_file) # will return None if model doesn't exist

    while True:
        _st = time.time()
        print("New Cycle")
        data = fetch_data(url)
        
        if data:
            model, anomalies = anomaly_detection(data, model=model) 
            if not anomalies.empty:
                send_email()
            save_model(model, model_pickle_file)
        
        else:
            print("Failed to retrieve data")
        
        _end = time.time()
        print("Waiting for the next cycle to begin.")
        time.sleep(max(100, _end - _st))

if __name__ == "__main__":
    main()
