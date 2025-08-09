from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import uuid
from flask import Flask, request, jsonify, send_from_directory

import os
import json
import asyncio
import pyaudio
import requests
from typing import Optional, Dict, Any
import pygame
import tempfile
from sarvamai.play import save
import google.generativeai as genai
from langdetect import detect
import logging
from sarvamai import SarvamAI
import uuid

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = 'static/temp' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
class VoiceAssistant:
    def __init__(self):
 
        self.sarvam_api_key =os.getenv("SARVAM_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        self.openai_client = None
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.client = SarvamAI(
            api_subscription_key=self.sarvam_api_key,)
        
        self.audio_config = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': 16000,
            'chunk': 1024
        }
        
   
        self.language_map = {
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'pa': 'Punjabi',
            'en': 'English'
        }
        
    
        self.sarvam_languages = ['hi', 'ta', 'te', 'kn', 'ml', 'bn', 'gu', 'mr', 'pa', 'en']
        
     
        pygame.mixer.init()
        
 
       
    def sarvam_stt(self, audio_file_path: str, language: str = 'auto') -> Optional[str]:
      
        if not self.sarvam_api_key:
            logger.warning("Sarvam API key not found")
            return None
        
        try:
            if(language=="auto"):
                response = self.client.speech_to_text.translate(
                    file=open(audio_file_path, "rb"),
                    model="saaras:v2.5"
                )
                print(type(response))
                print(response)
                transcript = response.transcript
                full_lang_code = response.language_code 
                lang_only = full_lang_code.split("-")[0]
                return transcript,lang_only
                
            response = self.client.speech_to_text.transcribe(
                file=open(audio_file_path, "rb"),
                model="saarika:v2.5",
                language_code=language
            )
            
            
            transcript = response.transcript
            print("Transcript:", transcript)
            
            return transcript
        except:
                    
                    return None
                    
       

    def detect_language(self, text: str) -> str:
      
        try:
            detected = detect(text)
            return detected if detected in self.language_map else 'en'
        except:
            return 'en'

    def generate_response_openai(self, text: str, language: str) -> str:
      
        if not self.openai_client:
            return "OpenAI client not initialized."
        
        try:
            language_name = self.language_map.get(language, 'English')
            prompt = f"Respond in {language_name} to the following: {text}"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. Always respond in {language_name}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return "Sorry, I couldn't process your request."

    def generate_response_gemini(self, text: str, language: str) -> str:
      
        try:
            language_name = self.language_map.get(language, 'English')
            prompt = f"Respond in {language_name} to the following query: {text}"
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if response.text else "Sorry, I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return "Sorry, I couldn't process your request."

    def sarvam_tts(self, text: str, language: str) -> Optional[str]:
        
        if not self.sarvam_api_key:
            return None
        
        try:
          
            unique_id = str(uuid.uuid4())
            input_filename = f"{unique_id}_recorded_audio.webm"
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            response = self.client.text_to_speech.convert(
                            text=text,
                            target_language_code=language,
                            model="bulbul:v2",
                            speaker="anushka"
                        )
           
            save(response, input_path)
            return input_path
            

          
        except Exception as e:
            logger.error(f"Error with Sarvam TTS: {e}")
            return None

  

    async def process_voice_input(self, audio_file, target_language: str = 'auto'):
       
        try:
            
            if target_language != 'auto' and target_language in self.sarvam_languages:
                lang=target_language+"-IN"
                transcribed_text = self.sarvam_stt(audio_file, lang)
            elif target_language=="auto":
                transcribed_text,target_language = self.sarvam_stt(audio_file, target_language)
                 
            if not transcribed_text:
                print("Could not transcribe audio")
                return
            
            print(f"Transcribed: {transcribed_text}")
            
            detected_language = self.detect_language(transcribed_text)
            language_to_use = target_language if target_language != 'auto' else detected_language
            
            print(f"Detected language: {self.language_map.get(language_to_use, 'Unknown')}")
            
            response_text = None
            
            if self.openai_client:
                response_text = self.generate_response_openai(transcribed_text, language_to_use)
            elif self.gemini_api_key:
                response_text = self.generate_response_gemini(transcribed_text, language_to_use)
            
            if not response_text:
                response_text = "Sorry, I couldn't generate a response."
            
            print(f"Response: {response_text}")
            
            audio_response_file = None
            
          
            if language_to_use in self.sarvam_languages:
                lang=language_to_use+"-IN"
                audio_response_file = self.sarvam_tts(response_text, lang)
            
            if audio_response_file:
               return audio_response_file

            else:
                print("Could not generate audio response")
            
        finally:
            if os.path.exists(audio_file):
                print("Completed")

    def interactive_mode(self,current_language,audio_file):
        result=asyncio.run(self.process_voice_input(audio_file,target_language=current_language))
        return result
              
assistant = VoiceAssistant()
@app.route('/get_audio/<filename>')
def get_audio(filename):
    return send_from_directory('static/temp', filename)

@app.route('/')
def index():
    """Renders the main page with the audio recorder."""
    return render_template('index.html')

@app.route('/save-record', methods=['POST'])
def save_record():
    """Handles the uploaded audio file and saves it to disk."""
    if 'audio_data' not in request.files:
        flash('No audio file part in the request', 'error')
        return redirect(url_for('index'))

    audio_file = request.files['audio_data']
    language = request.form['language']

    

    if 'audio_data' not in request.files:
        return 'No audio file uploaded', 400

    file = request.files['audio_data']
    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}_recorded_audio.webm"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)
    audio_file=input_path
    result=assistant.interactive_mode(language,audio_file)
    return result

   
if __name__ == '__main__':
    app.run(debug=False)