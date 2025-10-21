from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import re

# Setup Flask
app = Flask(__name__)
CORS(app)

# # Setup LangChain LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# Load model directly
from transformers import AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load model directly
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import json
import torch
import os
from datetime import datetime

# Get current datetime
now = datetime.now()
current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

with open("secrets.json") as f:
    data = json.load(f)
    google_api_key = data["api_key"]
    user = data["user"]
    password = data["password"]
    host = data["host"]
    port = data["port"]
    jwt_secret_key = data["jwt_secret_key"]


# set HUGGINGFACEHUB_API_TOKEN=api_key
os.environ["google_api_key"] = google_api_key
os.environ["user"] = user
os.environ["password"] = password
os.environ["host"] = host
os.environ["port"] = port



from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import psycopg2
from datetime import timedelta
import re

# PostgreSQL connection
def connect_to_db(dbname):
    conn = psycopg2.connect(
            dbname=dbname,
            user=os.environ["user"],
            password=os.environ["password"],
            host=os.environ["host"],
            port=os.environ["port"]
        )
    return conn


def create_tables():
    """ Create tables in the Dental DB database"""
    commands = (
        """CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",


        """CREATE TABLE IF NOT EXISTS appointments (
            appointment_id SERIAL PRIMARY KEY,
            user_id INT REFERENCES users(user_id),
            patient_name VARCHAR(100),
            contact_number VARCHAR(15),
            email VARCHAR(100),
            appointment_date DATE,
            appointment_time TIME,
            appointment_type VARCHAR(100),
            dental_concern VARCHAR(100),
            doctor_name VARCHAR(100),
            patient_status VARCHAR(100),
            insurance_provider VARCHAR(100),
            reason_for_visit VARCHAR(100),
            special_requests TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""
    )
    try:
        print("works")
        conn = connect_to_db(dbname="dental_db")
        with conn.cursor() as cur:
            # execute the CREATE TABLE statement
            for command in commands:
                cur.execute(command)
        conn.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

bcrypt = Bcrypt(app)
app.config['JWT_SECRET_KEY'] = jwt_secret_key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=2)
jwt = JWTManager(app)

# ---------------- USER AUTH ----------------

create_tables()

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

    conn = connect_to_db(dbname="dental_db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    if cur.fetchone():
        return jsonify({"error": "User already exists"}), 400

    cur.execute("INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)",
                (full_name, email, hashed_pw))
    conn.commit()
    return jsonify({"message": "User registered successfully"}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    conn = connect_to_db(dbname="dental_db")
    cur = conn.cursor()
    cur.execute("SELECT user_id, password_hash FROM users WHERE email = %s", (email,))
    user = cur.fetchone()

    if not user or not bcrypt.check_password_hash(user[1], password):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=user[0])
    return jsonify({"access_token": access_token}), 200

def save_to_db(data):
    try:
        conn = connect_to_db(dbname="dental_db")
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO appointments (
                patient_name, contact_number, email,
                appointment_date, appointment_time, appointment_type,
                dental_concern, doctor_name, patient_status, insurance_provider, 
                reason_for_visit, special_requests
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data.get("patient_name", ""),
            data.get("contact_number", ""),
            data.get("email", ""),
            data.get("preferred_date", None),
            data.get("preferred_time", None),
            data.get("appointment_type", ""),
            data.get("dental_concern", ""),
            data.get("doctor_name", ""),       
            data.get("patient_status", ""),     
            data.get("insurance_provider", ""),
            data.get("reason_for_visit", ""),
            data.get("special_requests", "")
        ))
        conn.commit()
    except Exception as e:
        print("DB insertion error:", e)
    finally:
        cur.close()
        conn.close()


# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# repo_id = "deepseek-ai/DeepSeek-R1-0528"

# Load model and tokenizer
# model_name = "microsoft/DialoGPT-medium"  # or mistralai/Mistral-7B-Instruct-v0.2
# print("Loading model... please wait (this might take a few minutes)...")

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )

chat_history = []  # memory is maintained outside the chain

# Initialize Google AI model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["google_api_key"]
)

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     max_length=128,
#     temperature=0.5,
#     huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
#     provider="auto",  # set your provider here hf.co/settings/inference-providers
#     # provider="hyperbolic",
#     # provider="nebius",
#     # provider="together",
# )

# ChatHuggingFace usage
# chat_model = ChatHuggingFace(llm=llm)

# @app.route("/api/chat", methods=["POST"])
# def chat():
#     user_message = request.json.get("message")
#     if not user_message:
#         return jsonify({"error": "Message required"}), 400

#     # Add user message to history
#     chat_history.append({"role": "user", "content": user_message})

#     # Combine all previous messages into one prompt
#     prompt = "You are a helpful assistant. Introduce yourself as an appointment scheduler. Provide answers in a clean and well-formatted style.\n"
#     # prompt = ""
#     for msg in chat_history[-5:]:
#         role = "User" if msg["role"]=="user" else "AI"
#         prompt += f"{role}: {msg['content']}\n"
#     prompt += "AI:"


#     # Tokenize and generate
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=300,
#         # temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     # Decode and clean response
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     response = response.split("AI:")[-1].strip()

#     # Add AI response to history
#     chat_history.append({"role": "assistant", "content": response})

#     return jsonify({"response": response})

@app.route("/api/chat", methods=["POST"])
def chat():
    # try:
    system_message = f"""You are a professional, friendly, and detail-oriented **dental appointment scheduling assistant** for a modern dental clinic.

    Current date and time: {current_time_str}

    Your role:
    - Engage patients naturally, asking one question at a time.
    - Collect all the following details in a structured, conversational way:
    1. Full Name of the patient
    2. Contact Number  
    3. Email (if available)  
    4. Preferred Date and Time for Appointment (Convert date and time mentioned to SQL-ready date and time format)
    5. Type of Appointment (e.g., Cleaning, Consultation, Filling, Root Canal, Tooth Extraction, Whitening, Braces Check-up, Emergency Visit)  
    6. Any Dental Concerns or Pain Level  
    7. Whether they are an existing or new patient  
    8. If they have Dental Insurance (if yes, request provider name)  
    9. Any Special Requests or Notes  
    10. Make up a doctor name, his qualifications and his expertise and assign him to the patient. 
    Mention the doctor's name and details to the patient.

    When all information is collected, summarize everything in this JSON format:
    {{
    "patient_name": "" (string),
    "contact_number": "" (string),
    "email": "" (string),
    "preferred_date": "" (date),
    "preferred_time": "" (time),
    "appointment_type": "" (string),
    "dental_concern": "" (string),
    "patient_status": "" (string),
    "insurance_provider": "" (string),
    "special_requests": "" (string),
    "doctor_name: "" (string),
    "reason_for_visit": "" (string)
    }}

    Guidelines:
    - Confirm details politely, in a well-structured manner and rephrase them for accuracy.
    - Never skip required information.
    - If information is missing, ask follow-up questions until all details are complete.
    - If the patient provides an invalid time (e.g., after clinic hours 9 AM–6 PM), politely ask for another slot.
    - Once all information is gathered, summarize everything in a well-structured format and ask for final confirmation.
    - Always remain professional, concise, and empathetic.

    Tone:
    - Friendly, warm, and reassuring — like a real receptionist at a high-quality dental clinic.
    - Avoid overly robotic or short responses.
    - Keep responses in short, clear paragraphs.

    You are not an AI chatbot — you represent the **Dental Clinic Appointment Desk**.
    """

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message required"}), 400

    # Create the prompt template
    prompt = PromptTemplate.from_template(
        (
            "[INST] {system_message}"
            "\nCurrent Conversation:\n{chat_history}\n\n"
            "\nUser: {user_message}.\n [/INST]"
            "\nAI:"
        )
    )

    # Make the chain and bind the prompt
    chat = prompt | chat_model | StrOutputParser(output_key='content')

    # Generate the response
    response = chat.invoke(
        input=dict(system_message=system_message, 
                   user_message=user_message, 
                   chat_history=chat_history)
    )
    # Only keep the newly generated response
    response = response.split("AI:")[-1]
    # Remove any <think>...</think> reasoning tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Update the chat history
    chat_history.append({'role': 'user', 'content': user_message})
    chat_history.append({'role': 'assistant', 'content': response})

    # Find all possible JSON-looking segments
    matches = re.findall(r'\{.*?\}', response, re.DOTALL)

    # Try extracting JSON if it exists
    if matches:
        try:
            data = json.loads(matches[-1])
            save_to_db(data)
        except Exception as e:
            print("Error parsing JSON:", e)
    else:
        print("No JSON found in model response")
    return jsonify({"response": response})
    

if __name__ == "__main__":
    # Make sure you set: export OPENAI_API_KEY="your-key"
    app.run(host="0.0.0.0", port=5000, debug=True)
