import fastapi
import whisper
import os
import speech_recognition as sr
from difflib import SequenceMatcher
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from gtts import gTTS
import time
import tempfile
import logging
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

app = fastapi.FastAPI()
model = whisper.load_model("base")
recognizer = sr.Recognizer()

# Included language information in the song dictionary
songs = {
    # English songs
    "song1": {"words": ["hello", "apple", "orange"], "lang": "en"},
    "song2": {"words": ["hello", "apple"], "lang": "en"},
    "twinkle": {"words": ["Twinkle", "twinkle", "little", "star", "how", "I", "wonder", "what", "you", "are"], "lang": "en"},
    "abc": {"words": ["a", "b", "c", "d", "e", "f", "g"], "lang": "en"},
    
    # Hindi songs
    "gayatri_mantra": {"words": ["ॐ", "भूर्भुवःस्वः", "तत्सवितुर्वरेण्यं", "भर्गो", "देवस्य", "धीमहि", "धियोयो", "नः", "प्रचोदयात्"], "lang": "hi"},
    "hindi_song": {"words": ["नमस्ते", "कैसे", "हो", "आप"], "lang": "hi"},
    
    # Marathi songs
    "marathi_song": {"words": ["नमस्कार", "कसे", "आहात"], "lang": "mr"}
}

# Thread-safe current_word_index dictionary
current_word_index = {song: 0 for song in songs}
index_lock = Lock()

# Currently active song
current_song = None
song_lock = Lock()

# Flag to control practice sessions
practice_running = False
practice_lock = Lock()

# Maximum number of attempts per word
MAX_ATTEMPTS = 5

def play_audio(text, lang='en'):
    """
    Generate and play audio using gTTS with language support
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            filename = temp_file.name
        
        # Generate speech with appropriate language
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        
        # Play the audio
        from playsound import playsound
        playsound(filename)
        
        # Remove the temporary audio file
        os.remove(filename)
    except Exception as e:
        logger.error(f"Error in play_audio: {e}")

def record_audio(duration=3, sample_rate=44100):
    """
    Record audio using sounddevice
    """
    logger.info("Recording...")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
        sd.wait()
        return recording
    except Exception as e:
        logger.error(f"Error in record_audio: {e}")
        return None

def save_audio(audio_data, filename=None):
    """
    Save numpy array audio data to wav file
    """
    if audio_data is None:
        return None
    
    try:
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                filename = temp_file.name
        
        sf.write(filename, audio_data, 44100)
        return filename
    except Exception as e:
        logger.error(f"Error in save_audio: {e}")
        return None

def pronounce_and_verify(song_name: str):
    global current_word_index, practice_running
    
    if song_name not in songs:
        return {"error": "Song not found"}
    
    song_data = songs[song_name]
    lang = song_data.get("lang", "en")
    words = song_data["words"]
    
    # Update current song
    with song_lock:
        global current_song
        current_song = song_name
    
    while True:
        # Check if practice should continue
        with practice_lock:
            if not practice_running:
                logger.info(f"Practice stopped for {song_name}")
                return {"status": "stopped", "song": song_name}
            
        # Get current word index in a thread-safe manner
        with index_lock:
            word_index = current_word_index[song_name]
            if word_index >= len(words):
                break
        
        word = words[word_index]
        logger.info(f"Current word: {word} (Language: {lang})")
        
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            # Check if practice should continue
            with practice_lock:
                if not practice_running:
                    logger.info(f"Practice stopped for {song_name}")
                    return {"status": "stopped", "song": song_name}
                
            # Speak the word with appropriate language
            play_audio(word, lang)
            time.sleep(0.5)  # Small delay to ensure audio is played before recording
            
            # Record user's response
            audio_data = record_audio()
            if audio_data is None:
                logger.warning("Failed to record audio. Retrying...")
                continue
                
            temp_file = save_audio(audio_data)
            if temp_file is None:
                logger.warning("Failed to save audio. Retrying...")
                continue
            
            try:
                # Transcribe the recorded audio
                result = model.transcribe(temp_file)
                recognized_text = result["text"].strip().lower()
                logger.info(f"Recognized: '{recognized_text}'")
                
                # Clean up temporary file
                os.remove(temp_file)
                
                # Check similarity
                similarity = similarity_ratio(word.lower(), recognized_text)
                logger.info(f"Similarity: {similarity}")
                
                if similarity >= 0.7:  # Slightly lower threshold for non-English languages
                    logger.info(f"Correct pronunciation of '{word}'")
                    # Update word index in thread-safe manner
                    with index_lock:
                        current_word_index[song_name] += 1
                    break
                else:
                    attempts += 1
                    logger.info(f"Incorrect pronunciation ({attempts}/{MAX_ATTEMPTS}). Repeating '{word}'")
                    # Speak feedback
                    if lang == "en":
                        play_audio("Incorrect pronunciation. Listen carefully.", "en")
                    elif lang == "hi":
                        play_audio("गलत उच्चारण। ध्यान से सुनिए।", "hi")
                    elif lang == "mr":
                        play_audio("चुकीचा उच्चार. काळजीपूर्वक ऐका.", "mr")
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                attempts += 1
        
        # If maximum attempts reached without success
        if attempts >= MAX_ATTEMPTS:
            logger.info(f"Maximum attempts reached for '{word}'. Moving to next word.")
            # Update word index in thread-safe manner
            with index_lock:
                current_word_index[song_name] += 1
    
    # Practice complete
    with practice_lock:
        practice_running = False
    
    logger.info(f"{song_name} completed")
    return {"status": "completed", "song": song_name}

@app.get("/progress/{song_name}")
def get_progress(song_name: str):
    if song_name not in songs:
        return {"error": "Song not found"}
    
    with index_lock:
        word_index = current_word_index[song_name]
        total_words = len(songs[song_name]["words"])
    
    # Get the actual words to provide more context
    words = songs[song_name]["words"]
    current_word = words[word_index] if word_index < total_words else "completed"
    
    # Get practice status
    with practice_lock:
        is_running = practice_running
        
    with song_lock:
        active_song = current_song
        
    is_active = active_song == song_name and is_running
    
    progress = word_index / total_words if total_words > 0 else 1.0
    return {
        "song": song_name, 
        "progress": progress, 
        "completed": word_index, 
        "total": total_words, 
        "current_word": current_word if word_index < total_words else None,
        "is_active": is_active,
        "running": is_running and active_song == song_name
    }

@app.get("/start/{song_name}")
def start_practice(song_name: str):
    global practice_running
    
    if song_name not in songs:
        return {"error": "Song not found"}
    
    # Stop any ongoing practice
    with practice_lock:
        practice_running = True
    
    # Reset the word index if requested
    with index_lock:
        current_word_index[song_name] = 0
    
    # Start the practice in a new thread
    threading.Thread(target=pronounce_and_verify, args=(song_name,), daemon=True).start()
    return {"message": "Practice started", "song": song_name, "language": songs[song_name]["lang"]}

@app.get("/stop")
def stop_practice():
    global practice_running
    
    with practice_lock:
        was_running = practice_running
        practice_running = False
    
    with song_lock:
        stopped_song = current_song
    
    return {"message": "Practice stopped", "was_running": was_running, "song": stopped_song}

@app.get("/available_songs")
def get_available_songs():
    result = {}
    for song_name, song_data in songs.items():
        result[song_name] = {
            "language": song_data["lang"],
            "word_count": len(song_data["words"]),
            "words": song_data["words"]  # Include the actual words in the response
        }
    return result

@app.get("/reset/{song_name}")
def reset_song_progress(song_name: str):
    if song_name not in songs:
        return {"error": "Song not found"}
    
    with index_lock:
        current_word_index[song_name] = 0
    
    return {"message": "Progress reset", "song": song_name}

@app.get("/song_details/{song_name}")
def get_song_details(song_name: str):
    if song_name not in songs:
        return {"error": "Song not found"}
    
    return {
        "song": song_name,
        "language": songs[song_name]["lang"],
        "words": songs[song_name]["words"],
        "word_count": len(songs[song_name]["words"])
    }

# Add CORS middleware to allow frontend requests
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# For running the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)