# Multimodal RAG Chatbot

An advanced multimodal Retrieval-Augmented Generation (RAG) chatbot that combines text retrieval using Pinecone and image retrieval using a Neo4j Knowledge Graph. 

The chatbot utilizes `LlamaCpp` (Mistral 7b) for text generation, `Pinecone` for vector storage of PDF contents, and `Neo4j` to maintain a knowledge graph of images with generated captions via `BLIP`.

## 📁 Repository Structure and Key Files

### `rag_chatbot.py`
This is the core setup and CLI script. It is responsible for:
1. **Text Pipeline**: Loading PDF documents (`Sample_RE.pdf`), splitting them into semantic chunks, and upserting the embeddings to a Pinecone vector index.
2. **Image Pipeline**: Reading continuous images (`.jfif`), generating image captions using Salesforce's BLIP model, and inserting these image nodes and their vector embeddings into a Neo4j Knowledge Graph.
3. **CLI Chat Loop**: A command-line interface to ask questions. It retrieves relevant text from Pinecone and links to relevant images from Neo4j to generate a grounded response using Mistral.

### `server.py`
This is the production-ready FastAPI backend that serves the RAG engine over HTTP.
- **FastAPI Setup**: Exposes the root `/` to serve the `run.html` frontend UI and serves static files (images) from the root directory.
- **RAGQueryEngine**: A class that initializes the LLM and vector stores upon server startup. It allows for parallel retrieval of context from both Pinecone and Neo4j to minimize latency.
- **API Endpoints**: Contains a `/query` POST endpoint which accepts user questions, retrieves multimodal context, and returns the AI-generated answer along with associated image URLs.

---

## 🚀 How to Run the Server Locally

To run the project on your localhost, you will use `uvicorn` to start the FastAPI server defined in `server.py`.

1. **Ensure your Virtual Environment is activated** (if you have one).
2. **Ensure your `.env` file is present** with all required keys (`PINECONE_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`).
3. **Start the server**:
   Run the following command in your terminal:
   ```bash
   uvicorn server:app --host 127.0.0.1 --port 8000 --reload
   ```
4. **Access the Chatbot**:
   Open your browser and navigate to: [http://127.0.0.1:8000](http://127.0.0.1:8000). You will be served the `run.html` frontend.

---

## 🐙 Pushing to GitHub

To build your new repository and push this project to GitHub under your username (`dheemanbabu`), follow these steps in your terminal:

1. **Initialize Git & Commit**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Multimodal RAG Chatbot"
   ```

2. **Create the Repository on GitHub**:
   *You can do this via the GitHub website (create a new empty repo named `multimodal_rag_chatbot`), or using the GitHub CLI (`gh`):*
   ```bash
   gh repo create multimodal_rag_chatbot --public --source=. --remote=origin
   ```

3. **Link and Push**:
   *(If you created the repo via the GitHub website instead of the CLI, link it first)*:
   ```bash
   git remote add origin https://github.com/dheemanbabu/multimodal_rag_chatbot.git
   ```
   *Then push your code:*
   ```bash
   git branch -M main
   git push -u origin main
   ```

Note: The included `.gitignore` will ensure that your `.env` file containing sensitive keys, IDE folders, and large `.gguf` local model files are not uploaded to GitHub.
