import streamlit as st
import requests
import time

BASE_URL = "http://localhost:8000"  # Change this if backend runs on a different host or port

st.set_page_config(page_title="Shlok Learning Platform", layout="centered")
st.title("Shlok Learning Platform 🗣️🎤")

# Fetch song list
@st.cache_data

def fetch_songs():
    try:
        res = requests.get(f"{BASE_URL}/available_songs")
        return res.json()
    except Exception as e:
        st.error("Error fetching songs")
        return {}

songs = fetch_songs()
song_names = list(songs.keys())

# Song selection
song_name = st.selectbox("Select a song to practice", song_names)
selected_song_data = songs[song_name]

# Display song details
with st.expander("📋 Song Details"):
    st.write(f"**Language:** {selected_song_data['language']}")
    st.write(f"**Words:** {' '.join(selected_song_data['words'])}")
    st.write(f"**Word Count:** {selected_song_data['word_count']}")

# State management
if 'running' not in st.session_state:
    st.session_state.running = False

# Start, Stop, Reset buttons
col1, col2, col3 = st.columns(3)

if col1.button("▶️ Start Practice"):
    response = requests.get(f"{BASE_URL}/start/{song_name}")
    if response.status_code == 200:
        st.session_state.running = True
        st.success("Practice started")

if col2.button("⏹ Stop Practice"):
    response = requests.get(f"{BASE_URL}/stop")
    if response.status_code == 200:
        st.session_state.running = False
        st.success("Practice stopped")

if col3.button("🔄 Reset Progress"):
    response = requests.get(f"{BASE_URL}/reset/{song_name}")
    if response.status_code == 200:
        st.success("Progress reset")

# Progress and dynamic update
progress_bar = st.progress(0)
current_word_display = st.empty()
stats_display = st.empty()

# Real-time progress update
if st.session_state.running:
    while st.session_state.running:
        progress_res = requests.get(f"{BASE_URL}/progress/{song_name}")
        if progress_res.status_code == 200:
            data = progress_res.json()
            progress = data['progress']
            progress_bar.progress(progress)

            word_list = songs[song_name]['words']
            current_index = data['completed']

            display_line = []
            for idx, word in enumerate(word_list):
                if idx == current_index:
                    display_line.append(f"**:orange[{word}]**")
                else:
                    display_line.append(word)

            current_word_display.markdown(" ".join(display_line))

            stats_display.info(f"Progress: {data['completed']}/{data['total']}")

            if data['running']:
                time.sleep(1.5)
            else:
                st.session_state.running = False
                st.success("🎉 Practice session completed!")
                break
        else:
            st.error("Error fetching progress")
            break
