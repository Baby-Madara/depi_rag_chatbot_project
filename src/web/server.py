import os
import json
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv

# Use ChatOllama for conversational memory compatibility
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# for fancy printing
from pprint import pprint
import json

# Load environment variables from .env
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

app = Flask(__name__, static_folder=base_dir, static_url_path='')

PORT = int(os.environ.get("PORT", 8000))
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- DATABASE (In-Memory for mock multiple users/chats) ---
'''
# Structure: { 
    user_id: 
    { 
        chat_id: { 
            "title": str, 
            "messages": [ 
                {"sender": "ai"/"user", "text": str} 
            ] 
        } 
    } 
}
'''
db = {
    "guest": {}
}

# --- LLM SETUP ---
if LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
else:
    llm = None

# --- ROUTES ---
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/config', methods=['GET'])
def get_config():
    frontend_config = {
        "companyName": os.environ.get("COMPANY_NAME", "AI Chatbot"),
        "theme": os.environ.get("THEME", "dark"),
        "enableSidebar": os.environ.get("ENABLE_SIDEBAR", "true").lower() == "true"
    }
    return jsonify(frontend_config)

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Returns a list of all chats for the user"""
    user_id = "guest" # Mock authentication
    user_chats = db.get(user_id, {})
    # Sort or just return as is. We'll return id and title.
    chats_list = [{"id": cid, "title": cinfo["title"]} for cid, cinfo in user_chats.items()]
    # Reverse so newest are first
    chats_list.reverse()
    return jsonify({"status": "success", "chats": chats_list})

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Creates a new chat session"""
    user_id = "guest"
    chat_id = str(uuid.uuid4())
    db[user_id][chat_id] = {
        "title": "New Chat",
        "messages": [
            {"sender": "ai", "text": "Hello! I am your AI Assistant. How can I help you today?"}
        ]
    }
    return jsonify({"status": "success", "chat_id": chat_id, "title": "New Chat"})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat_history(chat_id):
    """Retrieves full message history for a specific chat"""
    user_id = "guest"
    if chat_id not in db.get(user_id, {}):
        return jsonify({"status": "error", "error": "Chat not found"}), 404
    return jsonify({"status": "success", "messages": db[user_id][chat_id]["messages"]})


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming route with Context Awareness.
    Retrieves previous messages from db, passes to LangChain ChatOllama, and streams output.
    """
    data = request.get_json()
    user_message = data.get("message", "")
    chat_id = data.get("chat_id", "")
    user_id = "guest"

    if not chat_id or chat_id not in db.get(user_id, {}):
        return jsonify({"error": "Invalid chat_id"}), 400

    if not llm:
        return jsonify({"error": "LLM not configured"}), 400

    # Retrieve history
    history = db[user_id][chat_id]["messages"]
    
    # Auto-generate title if this is the first real message
    title_updated = False
    if db[user_id][chat_id]["title"] == "New Chat":
        db[user_id][chat_id]["title"] = user_message[:30] + ("..." if len(user_message) > 30 else "")
        title_updated = True

    # Save user message to history
    history.append({"sender": "user", "text": user_message})

    # Prepare Context-Aware LangChain Messages
    lc_messages = [
        SystemMessage(content=f"You are a helpful and professional assistant for {os.environ.get('COMPANY_NAME', 'our company')}.")
    ]
    
    # Inject up to the last 10 messages for context
    recent_history = history[-10:] if len(history) > 10 else history
    for msg in recent_history:
        if msg["sender"] == "user":
            lc_messages.append(HumanMessage(content=msg["text"]))
        elif msg["sender"] == "ai":
            lc_messages.append(AIMessage(content=msg["text"]))
    print(f"the conversation till now is: \n{json.dumps( [msg.dict() for msg in lc_messages], indent=4)}")

    def generate():
        ai_response_text = ""
        try:
            # Stream the response chunk by chunk from LangChain Chat Model
            for chunk in llm.stream(lc_messages):
                content = chunk.content
                if content:
                    ai_response_text += content
                    # Format as SSE
                    yield f"data: {json.dumps({'chunk': content})}\n\n"
            
            # Save AI message to history
            history.append({"sender": "ai", "text": ai_response_text})
            
            # If title was updated, notify frontend to update sidebar
            if title_updated:
                yield f"data: {json.dumps({'title': db[user_id][chat_id]['title']})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == "__main__":
    print(f"Starting Flask server on port {PORT} with LLM provider: {LLM_PROVIDER}")
    print(f"Make sure to install requirements using: pip install -r requirements.txt")
    app.run(host='0.0.0.0', port=PORT, debug=True)