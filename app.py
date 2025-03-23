import streamlit as st
import googlemaps
import folium
from streamlit_folium import folium_static
import cv2
import pyaudio
import wave
import threading
from datetime import datetime
from twilio.rest import Client
import json
from geopy import distance
import os
import openai
from PIL import Image
import numpy as np

# Load API keys securely
openai.api_key = os.getenv("OPENAI_API_KEY")
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(twilio_account_sid, twilio_auth_token)

class CrimeAnalyzer:
    def analyze_crime(self, video_path, audio_path):
        try:
            frames = self.extract_key_frames(video_path)
            frame_descriptions = []
            for frame in frames:
                img_str = self.encode_image(frame)
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "Describe any criminal activity in this image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]}
                    ]
                )
                frame_descriptions.append(response.choices[0].message.content)
            audio_context = self.analyze_audio(audio_path)
            return self.get_final_analysis(frame_descriptions, audio_context)
        except Exception as e:
            return f"Error in crime analysis: {str(e)}"

    def analyze_audio(self, audio_path):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript["text"]
        except Exception as e:
            return f"Error analyzing audio: {str(e)}"

    def extract_key_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        prev_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                if diff.mean() > 5:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            prev_frame = gray
        cap.release()
        return frames

    def encode_image(self, frame):
        import base64
        from io import BytesIO
        buffer = BytesIO()
        frame.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class SafetyApp:
    def check_location_safety(self, lat, lng):
        for zone in self.risk_zones["zones"]:
            dist = distance.distance((lat, lng), (zone["location"]["lat"], zone["location"]["lng"])).meters
            if dist <= self.alert_threshold:
                return {"safe": False, "message": f"Warning: High-risk area within {int(dist)}m."}
        return {"safe": True}

    def send_alert(self, msg):
        for contact in self.emergency_contacts:
            try:
                twilio_client.messages.create(
                    body=msg,
                    from_=os.getenv("TWILIO_PHONE_NUMBER"),
                    to=contact["phone"]
                )
            except Exception as e:
                print(f"Failed to send alert: {str(e)}")

# Streamlit UI
def main():
    st.title("Personal Safety Companion")
    app = SafetyApp()
    lat = st.number_input("Latitude", value=40.7128)
    lng = st.number_input("Longitude", value=-74.0060)
    safety_status = app.check_location_safety(lat, lng)
    if not safety_status["safe"]:
        st.warning(safety_status["message"])
    folium_static(folium.Map(location=[lat, lng], zoom_start=13))
    if st.button("🚨 Send Emergency Alert"):
        app.send_alert("Emergency! Crime detected!")
        st.success("Alert sent!")

if __name__ == "__main__":
    main()


