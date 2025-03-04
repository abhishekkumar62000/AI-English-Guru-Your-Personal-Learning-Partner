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
import subprocess
import spacy
import asyncio

# ✅ Fix: Ensure Spacy model is available
try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    spacy.load("en_core_web_sm")

# ✅ Fix: Event loop error in Streamlit Cloud
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# ✅ Fix: Load Gramformer Safely
try:
    from gramformer import Gramformer
    gf = Gramformer(models=1, use_gpu=False)
except Exception as e:
    st.error(f"⚠️ Error loading Gramformer: {e}")
    gf = None  # Prevents app from crashing

# ✅ Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API Key is missing! Please check your .env file.")
    st.stop()

# ✅ Fix: Gemini API Model Selection
try:
    model_name = "models/gemini-1.5-pro-latest"
    model = genai.GenerativeModel(model_name)
except Exception as e:
    st.error(f"⚠️ Error connecting to Gemini API: {e}")
    st.stop()

# ✅ AI Teacher Instructions
SYSTEM_PROMPT = """
You are an AI English Tutor. Follow these rules strictly:
1. Analyze the user's sentence and provide precise, high-quality corrections.
2. Avoid unnecessary fluff—be 100% to the point.
3. Keep responses engaging and interactive.
4. Provide practice exercises and quizzes.
"""

# ✅ Streamlit UI
st.set_page_config(page_title="AI English Guru", page_icon="🗣️", layout="wide")
st.title("🎙️ AI English Guru – Your Personal Learning Partner 🤖")
st.write("Improve your English step by step with interactive lessons!")

# ✅ Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ User Level Selection
user_level = st.sidebar.selectbox("Your Level", ["Beginner", "Intermediate", "Advanced"])
st.session_state["user_level"] = user_level

# ✅ Speech Recognition Setup
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

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

# ✅ Fix: Grammar & Spelling Correction
def correct_spelling(text):
    return str(TextBlob(text).correct())

def correct_grammar(text):
    if gf:
        corrected = list(gf.correct(text))
        return corrected[0] if corrected else text
    return text  # If Gramformer fails, return original text

# ✅ User Input Handling
if st.sidebar.button("🎤 Speak"):
    user_input = listen_speech()
    if user_input:
        st.text(f"You said: {user_input}")
else:
    user_input = st.chat_input("Ask your English learning question...")

if user_input:
    corrected_text = correct_spelling(user_input)
    grammatically_correct_text = correct_grammar(corrected_text)

    if grammatically_correct_text != user_input:
        st.write(f"🔍 Suggested Correction: {grammatically_correct_text}")

    st.session_state.chat_history.append({"role": "user", "content": grammatically_correct_text})

    # ✅ Optimize Chat Context
    chat_context = [SYSTEM_PROMPT]
    for chat in st.session_state.chat_history[-5:]:
        chat_context.append(f"{chat['role']}: {chat['content']}")
    chat_prompt = "\n".join(chat_context)

    # ✅ AI Response Generation
    try:
        response = model.generate_content(chat_prompt, generation_config={
            "temperature": 0.3,
            "top_k": 50,
            "top_p": 0.9,
            "max_output_tokens": 500
        })
        ai_reply = response.text
    except Exception as e:
        st.error(f"⚠️ Error generating AI response: {e}")
        ai_reply = "Sorry, I couldn't process that."

    st.session_state.chat_history.append({"role": "model", "content": ai_reply})

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

# ✅ Speak AI Response
if st.sidebar.button("🔊 Speak Response"):
    if st.session_state.chat_history:
        last_ai_response = st.session_state.chat_history[-1]["content"]
        speak_text(last_ai_response)
    else:
        st.sidebar.warning("No AI response to speak yet!")

# ✅ Grammar Quiz
quiz_questions = [
    {
        "question": "Which sentence is correct?",
        "options": ["She don't like apples.", "She doesn't likes apples.", "She doesn't like apples.", "She don't likes apples."],
        "answer": "She doesn't like apples."
    },
    {
        "question": "Choose the correct verb: 'He ____ to the gym every day.'",
        "options": ["go", "goes", "going", "gone"],
        "answer": "goes"
    }
]

def generate_quiz():
    return random.choice(quiz_questions)

if st.sidebar.button("📝 Take a Quiz"):
    question = generate_quiz()
    st.sidebar.markdown(f"**Question:** {question['question']}")
    user_answer = st.sidebar.radio("Options", question["options"])
    
    if st.sidebar.button("Submit Answer"):
        if user_answer == question["answer"]:
            st.sidebar.success("Correct! 🎉")
        else:
            st.sidebar.error(f"Incorrect. The correct answer is: {question['answer']}")

# ✅ Daily English Tip
english_tips = [
    "Read English books and articles daily.",
    "Practice speaking English with friends.",
    "Watch English movies with subtitles.",
    "Keep a journal in English to practice writing."
]

if st.sidebar.button("💡 Daily English Tip"):
    st.sidebar.markdown(f"**Tip of the Day:** {random.choice(english_tips)}")


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
