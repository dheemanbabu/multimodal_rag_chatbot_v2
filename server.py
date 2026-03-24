import os
import concurrent.futures
from dotenv import load_dotenv
from pydantic import BaseModel

# FastAPI Imports
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# LangChain / AI Imports
from langchain_classic.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from neo4j import GraphDatabase

# Load Env
load_dotenv()
set_llm_cache(InMemoryCache())


# 1. THE AI ENGINE 

class RAGQueryEngine:
    def __init__(self):
        print("--- Initializing AI Engine... ---")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize LLM
        self.llm = LlamaCpp(
            model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.1,
            max_tokens=512,
            n_ctx=2048,
            n_threads=6, 
            n_batch=512,
            verbose=False
        )

        # Initialize Pinecone
        self.vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"), 
            embedding=self.embeddings
        )

        # Initialize Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        print("--- Engine Ready ---")

    def _get_pinecone_data(self, query):
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            print(f"Pinecone Error: {e}")
            return ""

    def _get_neo4j_data(self, query_embedding):
        image_search_query = """
        CALL db.index.vector.queryNodes('photo_desc_index', 3, $embedding)
        YIELD node, score
        WHERE score > 0.4
        RETURN node.path AS path, node.description AS description
        """
        found_images = []
        image_descriptions = []
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(image_search_query, embedding=query_embedding)
                for record in result:
                    found_images.append(record["path"])
                    image_descriptions.append(record["description"])
        except Exception as e:
            print(f"Neo4j Error: {e}")
            
        desc_str = f"\nRelevant Image Descriptions: {', '.join(image_descriptions)}" if image_descriptions else ""
        return found_images, desc_str

    def get_response(self, query: str):
        # Generate embedding
        query_embedding = self.embeddings.embed_query(query)

        # Parallel Retrieval
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_pinecone = executor.submit(self._get_pinecone_data, query)
            future_neo4j = executor.submit(self._get_neo4j_data, query_embedding)

            context_text = future_pinecone.result()
            found_images, image_context_str = future_neo4j.result()

        # Generate LLM Response
        final_context = f"PDF Context:\n{context_text}\n{image_context_str}"
        
        template = """Use the context only below to answer the question. Do not make up information that is not in the context and use bullet points and emojis where necessary.\nContext: {context}\nQuestion: {question}\nAnswer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        
        result = llm_chain.invoke({"context": final_context, "question": query})
        return result['text'], found_images

# 2. FASTAPI SERVER SETUP
app = FastAPI()

# Mount the current directory  to access images

app.mount("/files", StaticFiles(directory="."), name="files")

# Global Engine Variable
engine = None

class QueryRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    """Initialize the heavy models when server starts"""
    global engine
    engine = RAGQueryEngine()

@app.get("/")
def read_root():
    """Serve the run.html file explicitly"""
    if os.path.exists("run.html"):
        return FileResponse("run.html")
    

@app.post("/query")
def run_query(request: QueryRequest):
    """Handle the user query"""
    if not engine:
        return {"error": "Engine is still loading..."}
    
    answer_text, image_paths = engine.get_response(request.text)
    
    # Convert local file paths to Web URLs
   
    web_image_urls = []
    if image_paths:
        for path in image_paths:
            filename = os.path.basename(path) # Get just "cat.jfif"
            web_image_urls.append(f"/files/{filename}")

    return {
        "answer": answer_text,
        "images": web_image_urls
    }