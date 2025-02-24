# Voice-Based Customer Service Assistant

A voice-enabled customer service system that can handle meeting scheduling, product inquiries, and general store information using speech recognition and text-to-speech capabilities.

## Core Features

### Voice Interaction
- Uses Deepgram for Speech-to-Text (STT)
- Uses ElevenLabs for Text-to-Speech (TTS)
- Handles continuous conversation until user says "exit" or "goodbye"

### Meeting Scheduling System
1. **Conversation State Tracking**
   - Maintains context of the conversation
   - Tracks selected staff member and scheduling progress

2. **Time Processing**
   - Extracts time from natural language input
   - Normalizes various time formats (e.g., "9pm", "9:00 PM", "nine in the evening")
   - Matches normalized times with staff availability

3. **Meeting Storage**
   - Generates unique meeting IDs
   - Stores meetings as JSON files in the meetings directory

### Data Management
- Loads store information from `description.json`
- Loads product catalog from `products.json`
- Loads staff details and availability from `staff.json`
- Maintains conversation history

## Key Components

### Time Handling
python
def normalize_time(self, time_str):
# Converts various time formats to standard "HH:MM AM/PM"
# Examples: "9pm" → "9:00 PM", "14:00" → "2:00 PM"
def extract_time_from_text(self, text):
# Extracts time from natural language sentences
# Example: "meeting at 9pm" → "9:00 PM"


### Meeting Scheduling Flow
1. User mentions "meeting"
2. System displays available staff
3. User specifies staff member
4. System shows available times
5. User specifies time
6. System confirms and stores meeting

## Future Improvements
- Complete other function like order and products

### Suggested Enhancements
1. **Name Matching**
   - Implement fuzzy string matching for staff names
   - Handle phonetic similarities (e.g., "Jackie"/"Jacky")

2. **Time Processing**
   - Add support for more time formats
   - Handle timezone specifications
   - Add meeting duration support

3. **Meeting Management**
   - Add conflict checking
   - Implement meeting modification
   - Add cancellation functionality

## Environment Setup
Required environment variables in `.env`:
- `ELEVEN_LABS_API_KEY`: For text-to-speech
- `ELEVEN_LABS_VOICE_ID`: Voice selection for TTS
- `OPENAI_API_KEY`: For natural language processing
- `DEEPGRAM_API_KEY`: For speech-to-text

## File Structure
├── app.py # Main application file
├── description.json # Store information
├── products.json # Product catalog
├── staff.json # Staff details and availability
├── meetings/ # Stored meeting records
├── responses/ # Voice response files
├── reviews/ # Customer review storage
├── audio_responses/ # Generated audio files
└── orders/ # Order records

```

This README provides an overview of the system's functionality, key components, and potential future improvements.