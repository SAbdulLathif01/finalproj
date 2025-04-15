from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import speech_recognition as sr
import pyttsx3

MODEL_NAME = "tiiuae/falcon-rw-1b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

def ask_chatbot(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.replace(user_input, "").strip()

# Speech-to-text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Speech recognition service is unavailable."

# Text-to-speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
