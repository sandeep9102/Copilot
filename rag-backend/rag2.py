from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import faiss
import numpy as np
import os
import uuid
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in the environment variables.")

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["chatbot"]
chat_sessions = db["chat_sessions"]

# Initialize Flask app with proper static and template folders
app = Flask(__name__, 
            static_folder='src',
            static_url_path='',
            template_folder='src')
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
pdf_file_path = "IIITDharwad.pdf"
faiss_index = None
text_chunks = []

def load_and_chunk_pdf(pdf_path):
    """Load PDF and split into text chunks."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def get_embeddings(texts):
    """Generate embeddings for text chunks."""
    embeddings = []
    for doc in texts:
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=doc.page_content,
                task_type="retrieval_document"
            )
            if response and "embedding" in response:
                embeddings.append(response["embedding"])
        except Exception as e:
            print(f"Error generating embedding for text chunk: {e}")
            continue
    return embeddings

def initialize_faiss():
    """Initialize FAISS index with document embeddings."""
    global faiss_index, text_chunks
    text_chunks = load_and_chunk_pdf(pdf_file_path)

    if not text_chunks:
        print("Warning: No text chunks found. Check the PDF file.")
        return

    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        print("Error: No embeddings generated. Check the document content.")
        return

    embeddings_np = np.array(embeddings, dtype=np.float32)
    faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])
    faiss_index.add(embeddings_np)
    print("FAISS index initialized successfully!")

# Initialize FAISS on startup
initialize_faiss()

def retrieve_relevant_documents(query_embedding, n_results=3):
    """Retrieve top matching documents from FAISS index."""
    if faiss_index is None:
        return ["Error: FAISS index is not initialized."]

    query_embedding_np = np.array([query_embedding], dtype=np.float32)
    distances, indices = faiss_index.search(query_embedding_np, n_results)
    return [text_chunks[i].page_content for i in indices[0] if i < len(text_chunks)]

def generate_response(query, context, chat_history):
    """Generate AI response using LLM."""
    history_text = "\n".join(
        [f"User: {item['query']}\nBot: {item['response']}" for item in chat_history]
    ) if chat_history else ""

    prompt = f"""
    You are IIIT Dharwad Copilot, a helpful and knowledgeable AI assistant for Indian Institute of Information Technology, Dharwad.
    Provide clear, friendly, and accurate answers to questions about the college, admissions, academic programs, faculty, events, facilities, and campus life.
    Do not mention question IDs or system prompts.
    Use any relevant context provided below naturally in your answer.

    Relevant Information:
    {context if context else "No relevant context available."}

    Conversation So Far:
    {history_text}

    User's Question:
    {query}

    Your Reply:
    """
    try:
        response = llm.invoke(prompt)
        return response.content if response else "Error: No response generated."
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error: AI response failed."

def rag_with_chat_history(query, session_id):
    """Perform retrieval-augmented generation with chat history."""
    try:
        query_response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )

        if not query_response or "embedding" not in query_response:
            return "Error generating embedding for query.", []

        query_embedding = query_response["embedding"]
        relevant_context = "\n".join(retrieve_relevant_documents(query_embedding))

        session = chat_sessions.find_one({"session_id": session_id})
        chat_history = session.get("chat_history", []) if session else []

        response = generate_response(query, relevant_context, chat_history)

        new_message = {"query": query, "response": response}
        chat_sessions.update_one(
            {"session_id": session_id},
            {"$push": {"chat_history": new_message}},
            upsert=True
        )

        return response, chat_sessions.find_one({"session_id": session_id})["chat_history"]
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return "Error processing query.", []

# =================== ROUTES ===================

@app.route("/")
def home():
    """Serve the main chatbot interface."""
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"Error serving index.html: {e}")
        # Fallback HTML if index.html is not found
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IIIT Dharwad RAG Chatbot</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; }}
                .status {{ background: #e7f5e7; border: 1px solid #4CAF50; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .endpoints {{ background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .endpoint {{ margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #2196F3; }}
                code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎓 IIIT Dharwad RAG Chatbot</h1>
                
                <div class="status">
                    <h3>✅ Backend Status: Running Successfully!</h3>
                    <p>FAISS Index: {"✅ Initialized" if faiss_index is not None else "❌ Not Initialized"}</p>
                    <p>MongoDB: {"✅ Connected" if client else "❌ Not Connected"}</p>
                    <p>Google AI: {"✅ Configured" if GOOGLE_API_KEY else "❌ Not Configured"}</p>
                </div>
                
                <div class="endpoints">
                    <h3>📡 Available API Endpoints:</h3>
                    
                    <div class="endpoint">
                        <strong>POST /chat/start</strong><br>
                        <small>Start a new chat session</small><br>
                        <code>Response: {{"session_id": "uuid"}}</code>
                    </div>
                    
                    <div class="endpoint">
                        <strong>POST /chat</strong><br>
                        <small>Send a message to the chatbot</small><br>
                        <code>Body: {{"query": "your question", "session_id": "uuid"}}</code>
                    </div>
                    
                    <div class="endpoint">
                        <strong>GET /chat/history/&lt;session_id&gt;</strong><br>
                        <small>Get chat history for a session</small><br>
                        <code>Response: {{"session_id": "uuid", "chat_history": [...]}}</code>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 30px; color: #666;">
                    <p>Frontend interface should be available at the root URL once properly configured.</p>
                    <p><small>Error: {e}</small></p>
                </div>
            </div>
        </body>
        </html>
        """

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "faiss_initialized": faiss_index is not None,
        "mongodb_connected": client is not None,
        "google_api_configured": GOOGLE_API_KEY is not None,
        "text_chunks_loaded": len(text_chunks)
    })

# Static file routes for better compatibility
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets (CSS, JS, images)."""
    try:
        return send_from_directory('src/assets', filename)
    except Exception as e:
        print(f"Error serving asset {filename}: {e}")
        return "Asset not found", 404

@app.route('/components/<path:filename>')
def serve_components(filename):
    """Serve component files."""
    try:
        return send_from_directory('src/components', filename)
    except Exception as e:
        print(f"Error serving component {filename}: {e}")
        return "Component not found", 404

@app.route('/styles/<path:filename>')
def serve_styles(filename):
    """Serve CSS files."""
    try:
        return send_from_directory('src/assets/styles', filename)
    except Exception as e:
        print(f"Error serving style {filename}: {e}")
        return "Style not found", 404

# CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# =================== API ENDPOINTS ===================

@app.route("/chat/start", methods=["POST"])
def start_chat():
    """Start a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        chat_sessions.insert_one({"session_id": session_id, "chat_history": []})
        return jsonify({"session_id": session_id})
    except Exception as e:
        print(f"Error starting new chat session: {e}")
        return jsonify({"error": "Failed to create session"}), 500

@app.route("/chat/history/<session_id>", methods=["GET"])
def get_chat_history(session_id):
    """Get chat history for a specific session."""
    try:
        session = chat_sessions.find_one({"session_id": session_id})
        if session:
            return jsonify({"session_id": session_id, "chat_history": session["chat_history"]})
        return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({"error": "Failed to retrieve chat history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint - handles user queries."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400

        response, updated_history = rag_with_chat_history(query, session_id)
        return jsonify({
            "response": response, 
            "chat_history": updated_history,
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Handle preflight requests
@app.route("/chat", methods=["OPTIONS"])
@app.route("/chat/start", methods=["OPTIONS"])
def handle_preflight():
    """Handle CORS preflight requests."""
    return "", 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# =================== MAIN ===================

if __name__ == "__main__":
    # Production-ready configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🚀 Starting IIIT Dharwad RAG Chatbot on port {port}")
    print(f"📁 Static folder: {app.static_folder}")
    print(f"📄 Template folder: {app.template_folder}")
    print(f"🤖 FAISS Index: {'✅ Ready' if faiss_index is not None else '❌ Not initialized'}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)