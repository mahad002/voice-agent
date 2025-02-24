import json
import uuid
from datetime import datetime
from pathlib import Path
import os
import requests
import speech_recognition as sr
import pygame
from dotenv import load_dotenv
import openai
import time
import logging
from deepgram import Deepgram

# Setup Logger
def make_logger(log_dir="logs", log_name="ovc", console_log=True):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s,%(name)s,%(levelname)s,%(message)s,%(details)s,%(further)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    if console_log:
        logger.addHandler(ch)

    return logger

# Initialize Logger
logger = make_logger(console_log=True)


class CustomerServiceSystem:
    def __init__(self):
        # Load environment variables
        load_dotenv(dotenv_path=Path('.') / '.env')

        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

        # Initialize API Clients
        self.openai_client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        self.deepgram = Deepgram(self.DEEPGRAM_API_KEY)

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Create directories
        for folder in ["responses", "reviews", "audio_responses", "orders", "meetings"]:
            Path(folder).mkdir(exist_ok=True)

        # Load store description
        self.store_info = self.load_json_file('description.json', default={
            "store_name": "Unknown Store",
            "store_description": "No description available.",
            "product_categories": []
        })
        logger.info("Store Loaded", extra={"details": self.store_info['store_name'], "further": ""})

        # Load product catalog
        self.products = self.load_json_file('products.json', default=[])
        logger.info("Products Loaded", extra={"details": f"{len(self.products)} products", "further": ""})

        # Load staff details
        self.staff = self.load_json_file('staff.json', default=[])
        logger.info("Staff Loaded", extra={"details": f"{len(self.staff)} staff members", "further": ""})

        # Context tracking
        self.conversation_history = []

    def load_json_file(self, filename, default):
        """Load JSON file or return default if not found."""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"{filename} not found. Using default values.")
            return default

    def get_voice_input(self):
        """Capture and transcribe speech using Deepgram STT."""
        logger.info("Listening...", extra={"details": "User speaking", "further": ""})

        try:
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)

            # Convert audio to bytes
            audio_data = audio.get_wav_data()

            # Send to Deepgram for transcription
            response = self.deepgram.transcription.sync_prerecorded(
                {"buffer": audio_data, "mimetype": "audio/wav"},
                {"model": "nova-2", "language": "en", "smart_format": True}
            )

            if 'results' in response and 'channels' in response['results']:
                text = response['results']['channels'][0]['alternatives'][0]['transcript']
                logger.info("STT Success", extra={"details": text, "further": ""})
                return text.lower()
            else:
                logger.warning("No transcription result.")
                return ""

        except Exception as e:
            logger.error("STT Error", extra={"details": str(e), "further": ""})
            return ""

    def generate_speech(self, text):
        """Generate speech using Eleven Labs API and return audio file path."""
        logger.info("Generating Speech", extra={"details": text, "further": ""})

        print("ELEVENLABS_VOICE_ID", self.ELEVENLABS_VOICE_ID)
        print("ELEVENLABS_API_KEY", self.ELEVENLABS_API_KEY)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                filename = f"audio_responses/response_{int(time.time())}.mp3"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
            else:
                logger.error("TTS API Error", extra={"details": response.status_code, "further": ""})
                return None
        except Exception as e:
            logger.error("TTS Error", extra={"details": str(e), "further": ""})
            return None

    def play_audio(self, audio_file):
        """Play the generated audio response using pygame."""
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
        except Exception as e:
            logger.error("Audio Playback Error", extra={"details": str(e), "further": ""})

    def normalize_time(self, time_str):
        """Convert various time formats to a standard format (HH:MM AM/PM)"""
        time_str = time_str.lower().strip()
        
        # Remove dots and extra spaces
        time_str = time_str.replace('.', '').replace('  ', ' ')
        
        # Handle "10am" -> "10:00 AM" format
        if 'am' in time_str or 'pm' in time_str:
            # Remove spaces before am/pm
            time_str = time_str.replace(' am', 'am').replace(' pm', 'pm')
            
            # Extract hours
            hours = time_str.replace('am', '').replace('pm', '').strip()
            if ':' not in hours:
                hours = f"{hours}:00"
            
            # Add proper spacing and capitalization for AM/PM
            if 'am' in time_str:
                return f"{hours} AM"
            return f"{hours} PM"
        
        # Handle "10" -> "10:00 AM" format
        if time_str.isdigit():
            hour = int(time_str)
            if hour < 12:
                return f"{hour}:00 AM"
            elif hour == 12:
                return "12:00 PM"
            else:
                return f"{hour-12}:00 PM"
        
        return time_str.upper()

    def extract_time_from_text(self, text):
        """Extract time from a natural language sentence and normalize it."""
        text = text.lower()
        words = text.split()
        
        # Look for time patterns in the text
        for i, word in enumerate(words):
            # Check for patterns like "9pm", "9 pm", "9:00pm", "9:00 pm"
            if any(period in word for period in ['am', 'pm']):
                return self.normalize_time(word)
            
            # Check for patterns like "at 9" or "for 9"
            if word in ['at', 'for', 'around'] and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word.replace(':', '').replace('.', '').isdigit():
                    # Check if AM/PM is in the following word
                    if i + 2 < len(words) and any(period in words[i + 2].lower() for period in ['am', 'pm']):
                        return self.normalize_time(f"{next_word} {words[i + 2]}")
                    else:
                        return self.normalize_time(next_word)
        
        return None

    def handle_query(self, user_input):
        """Process user input and generate meaningful responses dynamically."""
        # Track conversation state
        if not hasattr(self, 'conversation_state'):
            self.conversation_state = {'state': None, 'data': {}}
        
        # Initial meeting request
        if "meeting" in user_input and self.conversation_state['state'] is None:
            staff_names = ", ".join([s["name"] for s in self.staff])
            self.conversation_state['state'] = 'meeting_requested'
            return f"Available staff members are: {staff_names}. Please specify who you'd like to meet with and your preferred time."
        
        # Handle meeting details
        if self.conversation_state['state'] == 'meeting_requested':
            # If we already have a staff member selected, look for time
            if 'selected_staff' in self.conversation_state['data']:
                matched_staff = self.conversation_state['data']['selected_staff']
                matched_time = None
                
                # Extract and normalize time from the sentence
                extracted_time = self.extract_time_from_text(user_input)
                if extracted_time:
                    # Try to match with available times
                    for time in matched_staff['availability']:
                        if self.normalize_time(time) == extracted_time:
                            matched_time = time
                            break
                
                if matched_time:
                    # Store meeting
                    meeting_id = str(uuid.uuid4())
                    meeting_data = {
                        'id': meeting_id,
                        'staff': matched_staff['name'],
                        'time': matched_time,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Save meeting to file
                    meetings_file = Path('meetings') / f'{meeting_id}.json'
                    with open(meetings_file, 'w') as f:
                        json.dump(meeting_data, f)
                    
                    # Reset conversation state
                    self.conversation_state = {'state': None, 'data': {}}
                    
                    return f"Great! I've scheduled your meeting with {matched_staff['name']} at {matched_time}."
                else:
                    available_times = ", ".join(matched_staff['availability'])
                    return f"I couldn't match that time. Available times for {matched_staff['name']} are: {available_times}. Please specify your preferred time."
            
            # If no staff selected yet, try to match staff name
            else:
                for staff in self.staff:
                    if staff['name'].lower() in user_input.lower():
                        self.conversation_state['data']['selected_staff'] = staff
                        available_times = ", ".join(staff['availability'])
                        return f"I found {staff['name']}. Their available times are: {available_times}. Please specify your preferred time."
                
                staff_names = ", ".join([s["name"] for s in self.staff])
                return f"I couldn't find that staff member. Available staff are: {staff_names}. Please try again."

        if "about your brand" in user_input or "tell me about" in user_input:
            return self.store_info['store_description']

        if "product" in user_input or "list" in user_input:
            product_list = ", ".join([p['name'] for p in self.products])
            return f"We have: {product_list}. Which one interests you?"

        if "order" in user_input:
            return "I can assist you in placing an order. What would you like to buy?"

        if "return" in user_input:
            return "Would you like to return a product? Please provide the name."

        return "I'm not quite sure about that. Can you clarify?"

    def run_conversation(self):
        """Main voice-based conversation loop with the customer."""
        logger.info("Starting conversation...", extra={"details": "", "further": ""})

        greeting = f"Welcome to {self.store_info['store_name']}! How may I assist you today?"
        print(f"AI: {greeting}")
        audio_file = self.generate_speech(greeting)
        if audio_file:
            self.play_audio(audio_file)

        while True:
            user_input = self.get_voice_input()
            if not user_input:
                response = "I didn't catch that, could you repeat?"
                logger.warning("No User Input")
                print(f"AI: {response}")
                audio_file = self.generate_speech(response)
                if audio_file:
                    self.play_audio(audio_file)
                continue

            if "exit" in user_input or "goodbye" in user_input:
                farewell = "Goodbye! Have a great day!"
                print(f"AI: {farewell}")
                audio_file = self.generate_speech(farewell)
                if audio_file:
                    self.play_audio(audio_file)
                break

            response = self.handle_query(user_input)
            print(f"AI: {response}")
            audio_file = self.generate_speech(response)
            if audio_file:
                self.play_audio(audio_file)


if __name__ == "__main__":
    system = CustomerServiceSystem()
    system.run_conversation()
