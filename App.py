import spacy
import subprocess

try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    spacy.load("en_core_web_sm")



# Fix event loop error
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Fix event loop error

from gramformer import Gramformer

gf = Gramformer(models=1, use_gpu=False)  # Now it should work

import streamlit as st
import google.generativeai as genai
import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from dotenv import load_dotenv
from textblob import TextBlob
import time
from io import BytesIO
import base64
import random
import difflib
import json

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API Key is missing! Please check your .env file or environment variables.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Fix: Ensure the correct model is used
try:
    model_name = "models/gemini-1.5-pro-latest"  # Best available model
    available_models = [m.name for m in genai.list_models()]
    
    if model_name not in available_models:
        st.error(f"⚠️ Model '{model_name}' not found! Available models: {available_models}")
        st.stop()

    model = genai.GenerativeModel(model_name)
except Exception as e:
    st.error(f"⚠️ Error connecting to Gemini API: {e}")
    st.stop()

# AI Teacher Instructions
SYSTEM_PROMPT = """
You are an AI English Tutor.Follow these rules strictly, Your task is to teach spoken English in a structured way.
0. Communicate with the user in a friendly and engaging manner.
1. Analyze the user's sentence, talk to User and perform Conversation and Provide Feedback.
2. Analyze the user's sentence precise, and high-quality answers..
3. If there is a mistake, correct it.
4. Avoid unnecessary fluff—be 100% to the point, Explain why the correction was needed..
5. Keep the response engaging and interactive.
5. Provide an example sentence.
6. If the user is a beginner, keep explanations simple.
7. If the user is advanced, give detailed grammar rules.
8. Provide practice exercises and quizzes.
9. Maintain a conversational and engaging tone.
"""

# Streamlit UI Design
st.set_page_config(page_title="AI English Teacher", page_icon="🗣️", layout="wide")

st.title("🎙️AI English Guru–Your Personal Learning Partner🤖")
st.write("Improve your English step by step with interactive lessons!")

AI_path = "AI.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(AI_path)
except FileNotFoundError:
    st.sidebar.warning("AI.png file not found. Please check the file path.")

image_path = "image.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(image_path)
except FileNotFoundError:
    st.sidebar.warning("image.png file not found. Please check the file path.")

# Add Developer Information to Sidebar
st.sidebar.markdown("👨👨‍💻Developer:- Abhishek❤️Yadav")

developer_path = "my.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(developer_path)
except FileNotFoundError:
    st.sidebar.warning("my.jpg file not found. Please check the file path.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_level" not in st.session_state:
    st.session_state.user_level = "Beginner"

# Select User Level
user_level = st.sidebar.selectbox("Your Level", ["Beginner", "Intermediate", "Advanced"])
st.session_state.user_level = user_level

# Initialize session state for progress tracking
if "progress" not in st.session_state:
    st.session_state.progress = {"grammar": 0, "vocabulary": 0, "pronunciation": 0}

# Function to update progress
def update_progress(category, points):
    st.session_state.progress[category] += points
    with open("progress.json", "w") as f:
        json.dump(st.session_state.progress, f)

# Load progress from file
if os.path.exists("progress.json"):
    with open("progress.json", "r") as f:
        st.session_state.progress = json.load(f)

# Display progress
st.sidebar.markdown("### Progress")
st.sidebar.write(f"Grammar: {st.session_state.progress['grammar']} points")
st.sidebar.write(f"Vocabulary: {st.session_state.progress['vocabulary']} points")
st.sidebar.write(f"Pronunciation: {st.session_state.progress['pronunciation']} points")

# Function to evaluate user input and update progress
def evaluate_and_update_progress(user_input, corrected_text, grammatically_correct_text):
    if user_input != corrected_text:
        update_progress("vocabulary", 5)
    if corrected_text != grammatically_correct_text:
        update_progress("grammar", 5)
    # Add more criteria as needed

# Speech Recognition
recognizer = sr.Recognizer()

def listen_speech():
    try:
        with sr.Microphone() as source:
            st.write("🎤 Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand."
    except sr.RequestError:
        return "Speech Recognition service unavailable."
    except Exception as e:
        st.error(f"Microphone Error: {e}")
        return None

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

def save_audio(text):
    try:
        tts = gTTS(text)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_base64 = base64.b64encode(audio_fp.read()).decode()
        return audio_base64
    except Exception as e:
        st.error(f"Error in saving audio: {e}")
        return None

# Function to get pronunciation feedback
def get_pronunciation_feedback(user_text, correct_text):
    user_words = user_text.split()
    correct_words = correct_text.split()
    diff = difflib.ndiff(correct_words, user_words)
    feedback = []
    for word in diff:
        if word.startswith('-'):
            feedback.append(f"Missed word: {word[2:]}")
        elif word.startswith('+'):
            feedback.append(f"Extra word: {word[2:]}")
        elif word.startswith('?'):
            feedback.append(f"Pronunciation issue: {word[2:]}")
    return feedback

# Speech Button
if st.sidebar.button("🎤 Speak"):
    user_input = listen_speech()
    if user_input:
        st.text(f"You said: {user_input}")
else:
    user_input = st.chat_input("Ask your English learning question...")

# Grammar & Spelling Correction
gf = Gramformer(models=1, use_gpu=False)

def correct_grammar(text):
    corrected = list(gf.correct(text))
    return corrected[0] if corrected else text

def correct_spelling(text):
    return str(TextBlob(text).correct())

if user_input:
    corrected_text = correct_spelling(user_input)
    grammatically_correct_text = correct_grammar(corrected_text)

    if grammatically_correct_text != user_input:
        st.write(f"🔍 Suggested Correction: {grammatically_correct_text}")

    st.session_state.chat_history.append({"role": "user", "content": grammatically_correct_text})

    with st.chat_message("user"):
        st.markdown(grammatically_correct_text)

    # Evaluate user input and update progress
    evaluate_and_update_progress(user_input, corrected_text, grammatically_correct_text)

    # Optimize Chat Context to Keep AI Focused
    chat_context = [SYSTEM_PROMPT]
    for chat in st.session_state.chat_history[-5:]:  # Only last 5 messages for better accuracy
        chat_context.append(f"{chat['role']}: {chat['content']}")

    chat_prompt = "\n".join(chat_context)

    # AI generation settings for high-quality answers
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(chat_prompt, generation_config={
            "temperature": 0.3,  # Lower = More precise responses
            "top_k": 50,         # Ensures high-quality tokens
            "top_p": 0.9,        # Balanced sampling
            "max_output_tokens": 500  # Limits overly long responses
        })
        ai_reply = response.text
    except Exception as e:
        st.error(f"⚠️ Error generating AI response: {e}")
        ai_reply = "Sorry, I couldn't process that."

    st.session_state.chat_history.append({"role": "model", "content": ai_reply})

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

# Speak AI Response
if st.sidebar.button("🔊 Speak Response"):
    if st.session_state.chat_history:
        last_ai_response = st.session_state.chat_history[-1]["content"]
        speak_text(last_ai_response)
    else:
        st.sidebar.warning("No AI response to speak yet!")

# Pronunciation Practice
if st.sidebar.button("🔊 Pronunciation Practice"):
    st.sidebar.write("Please pronounce the following sentence:")
    practice_sentence = "The quick brown fox jumps over the lazy dog."
    st.sidebar.write(f"**Sentence:** {practice_sentence}")
    
    if st.sidebar.button("🎤 Record Pronunciation"):
        user_pronunciation = listen_speech()
        if user_pronunciation:
            st.sidebar.write(f"You said: {user_pronunciation}")
            feedback = get_pronunciation_feedback(user_pronunciation, practice_sentence)
            if feedback:
                st.sidebar.write("**Feedback:**")
                for item in feedback:
                    st.sidebar.write(f"- {item}")
            else:
                st.sidebar.write("Great job! Your pronunciation is perfect.")

# Grammar Quiz Questions
quiz_questions = [
    {
        "question": "Which sentence is correct?",
        "options": ["She don't like apples.", "She doesn't likes apples.", "She doesn't like apples.", "She don't likes apples."],
        "answer": "She doesn't like apples."
    },
    {
        "question": "Choose the correct form of the verb: 'He ____ to the gym every day.'",
        "options": ["go", "goes", "going", "gone"],
        "answer": "goes"
    },
    {
        "question": "Which word is a noun?",
        "options": ["quickly", "run", "happiness", "blue"],
        "answer": "happiness"
    }
]

def generate_quiz():
    question = random.choice(quiz_questions)
    return question

# Add Grammar Quiz to Sidebar
if st.sidebar.button("📝 Take a Grammar Quiz"):
    question = generate_quiz()
    st.sidebar.markdown(f"**Question:** {question['question']}")
    user_answer = st.sidebar.radio("Options", question["options"])
    
    if st.sidebar.button("Submit Answer"):
        if user_answer == question["answer"]:
            st.sidebar.success("Correct! 🎉")
            update_progress("grammar", 10)
        else:
            st.sidebar.error(f"Incorrect. The correct answer is: {question['answer']}")

# Practice Quiz Feature
def generate_quiz():
    quiz_prompt = "Generate a simple English learning MCQ quiz with 4 options and 1 correct answer."
    try:
        response = model.generate_content(quiz_prompt)
        return response.text
    except Exception as e:
        st.error(f"⚠️ Error generating quiz: {e}")
        return "Quiz generation failed."

if st.sidebar.button("📖 Take a Quiz"):
    quiz = generate_quiz()
    st.markdown(quiz)

# Feedback System
feedback = st.sidebar.radio("How was the AI's response?", ["👍 Helpful", "🤔 Needs Improvement", "👎 Not Good"])
if feedback == "👍 Helpful":
    st.sidebar.success("Glad it helped! 😊")
elif feedback == "🤔 Needs Improvement":
    st.sidebar.warning("Thanks! We'll work on making it better. 🚀")
elif feedback == "👎 Not Good":
    st.sidebar.error("Sorry! We'll improve the AI. 😓")

# Vocabulary Dictionary
vocabulary = {
    "abate": {
        "meaning": "to lessen in intensity or degree",
        "example": "The storm suddenly abated."
    },
    "benevolent": {
        "meaning": "well-meaning and kindly",
        "example": "She was a benevolent woman, volunteering all of her free time to charitable organizations."
    },
    "candid": {
        "meaning": "truthful and straightforward",
        "example": "His responses were remarkably candid."
    },
    "diligent": {
        "meaning": "having or showing care in one's work or duties",
        "example": "He was a diligent student, always completing his assignments on time."
    },
    "emulate": {
        "meaning": "to match or surpass, typically by imitation",
        "example": "She tried to emulate her mentor's success."
    }
}

def get_daily_word():
    word, details = random.choice(list(vocabulary.items()))
    return word, details["meaning"], details["example"]

# Add Vocabulary Builder to Sidebar
if st.sidebar.button("📚 Daily Vocabulary"):
    word, meaning, example = get_daily_word()
    st.sidebar.markdown(f"**Word of the Day:** {word}\n\n**Meaning:** {meaning}\n\n**Example Sentence:** {example}")

# List of English Tips
english_tips = [
    "Read English books, newspapers, and articles to improve your reading skills.",
    "Practice speaking English with friends or language partners.",
    "Watch English movies and TV shows to improve your listening skills.",
    "Keep a journal in English to practice writing.",
    "Learn new vocabulary words every day and use them in sentences.",
    "Practice pronunciation by listening to native speakers and repeating after them.",
    "Use language learning apps to practice grammar and vocabulary.",
    "Join an English language club or group to practice speaking with others.",
    "Take online English courses to improve your skills.",
    "Set specific goals for your English learning and track your progress."
]

def get_daily_tip():
    tip = random.choice(english_tips)
    return tip

# Add Daily English Tip to Sidebar
if st.sidebar.button("💡 Daily English Tip"):
    tip = get_daily_tip()
    st.sidebar.markdown(f"**Tip of the Day:** {tip}")

# Conversation Prompts
conversation_prompts = [
    "Hi! How are you today? 😊",
    "What did you do over the weekend? 🌞",
    "Tell me about your favorite hobby. 🎨",
    "What are your plans for the upcoming holiday? 🎉",
    "Describe your favorite movie and why you like it. 🎬"
]

def start_conversation():
    prompt = random.choice(conversation_prompts)
    return prompt

# Function to add emojis to AI responses
def add_emojis_to_response(response):
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "excited": "🎉",
        "love": "❤️",
        "great": "👍",
        "good": "🙂",
        "bad": "🙁",
        "movie": "🎬",
        "hobby": "🎨",
        "holiday": "🎉",
        "weekend": "🌞"
    }
    for word, emoji in emoji_map.items():
        response = response.replace(word, f"{word} {emoji}")
    return response

# Add Conversation Practice to Sidebar
if st.sidebar.button("💬 Start Conversation Practice"):
    prompt = start_conversation()
    st.sidebar.markdown(f"**AI:** {prompt}")
    user_response = st.sidebar.text_input("Your response:")
    
    if user_response:
        st.sidebar.markdown(f"**You:** {user_response}")
        
        # Generate AI response
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"AI: {prompt}\nUser: {user_response}\nAI:", generation_config={
                "temperature": 0.3,  # Lower = More precise responses
                "top_k": 50,         # Ensures high-quality tokens
                "top_p": 0.9,        # Balanced sampling
                "max_output_tokens": 500  # Limits overly long responses
            })
            ai_reply = response.text
            ai_reply_with_emojis = add_emojis_to_response(ai_reply)
        except Exception as e:
            st.error(f"⚠️ Error generating AI response: {e}")
            ai_reply_with_emojis = "Sorry, I couldn't process that. 😓"
        
        st.sidebar.markdown(f"**AI:** {ai_reply_with_emojis}")
