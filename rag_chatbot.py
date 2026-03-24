import os
import glob
import time
from typing import List

# --- Libraries for PDF & Text RAG ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

# --- Libraries for Image & Knowledge Graph ---
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- OFFICIAL NEO4J DRIVER ---
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Neo4j Credentials from your .env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MODEL_PATH = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf" 
PDF_PATH = "Sample_RE.pdf"
IMAGE_FOLDER = "." 

# Embedding Model (Dimensions: 384)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# 2. SETUP MODELS
# ==========================================
print("--- Loading Models ---")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Image Captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_tokens=512,
    n_ctx=2048,
    verbose=False
)


# 3. TEXT PIPELINE (PDF -> Pinecone)

def setup_text_rag():
    print(f"--- Processing PDF: {PDF_PATH} ---")
    
    # Check if PDF exists to avoid errors
    if not os.path.exists(PDF_PATH):
        print(f"Warning: {PDF_PATH} not found. Skipping PDF processing.")
        return None

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    print("--- Creating Semantic Chunks ---")
    text_splitter = SemanticChunker(embeddings)
    docs = text_splitter.split_documents(documents)
    
    print("--- Upserting to Pinecone ---")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    vectorstore = PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name=PINECONE_INDEX_NAME
    )
    return vectorstore


# 4. IMAGE PIPELINE (Images -> Neo4j KG)

def generate_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = vision_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def setup_image_knowledge_graph():
    print("--- Building Knowledge Graph for Images ---")
    
    # 1. Connect using the official Driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # 2. Define Cypher queries
    # Query to create Vector Index 
    create_index_query = """
    CREATE VECTOR INDEX photo_desc_index IF NOT EXISTS
    FOR (p:Photo)
    ON (p.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 384,
      `vector.similarity_function`: 'cosine'
    }}
    """

    # Query to Create Node
    create_node_query = """
    MERGE (p:Photo {path: $path})
    SET p.description = $desc,
        p.embedding = $embedding
    """

    with driver.session() as session:
        # A. Clear existing data 
        session.run("MATCH (n) DETACH DELETE n")
        
        # B. Create the Vector Index
        session.run(create_index_query)
        print("Vector Index created/verified.")

        # C. Process Images
        image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.jfif"))
        
        for img_path in image_files:
            caption = generate_caption(img_path)
            if caption:
                print(f"Encoded {img_path}: {caption}")
                
                # Generate Embedding for the caption
                caption_embedding = embeddings.embed_query(caption)
                
                # Run Cypher to insert node
                session.run(create_node_query, 
                            path=img_path, 
                            desc=caption, 
                            embedding=caption_embedding)

    return driver  

# 5. RETRIEVAL & GENERATION

def chat_loop(text_vectorstore, neo4j_driver):
    print("\n" + "="*50)
    print("SYSTEM READY. Ask a question about the PDF or Images.")
    print("="*50 + "\n")

    while True:
        query = input("User Query: ")
        if query.lower() in ['exit', 'quit']:
            break

        # Part A: Text Retrieval (Pinecone) 
        context_text = ""
        if text_vectorstore:
            docs = text_vectorstore.similarity_search(query, k=3)
            context_text = "\n".join([d.page_content for d in docs])

        #  Part B: Image Retrieval (Neo4j Driver) 
        query_embedding = embeddings.embed_query(query)
        
        # Raw Cypher for Vector Search
        image_search_query = """
        CALL db.index.vector.queryNodes('photo_desc_index', 3, $embedding)
        YIELD node, score
        WHERE score > 0.5
        RETURN node.path AS path, node.description AS description, score
        """
        
        found_images = []
        image_context_str = ""
        
        # Execute Query using Driver
        with neo4j_driver.session() as session:
            result = session.run(image_search_query, embedding=query_embedding)
            records = list(result)
            
            if records:
                found_images = [record["path"] for record in records]
                descriptions = [record["description"] for record in records]
                image_context_str = f"\nRelevant Image Descriptions found: {', '.join(descriptions)}"

        # Part C LLM Generation
        final_context = f"PDF Context:\n{context_text}\n{image_context_str}"
        
        template = """You are a helpful assistant. Use the following context to answer the user's question.
        
        Context:
        {context}
        
        Question: 
        {question}
        
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        
        print("\n--- AI Response ---")
        response = llm_chain.invoke({"context": final_context, "question": query})
        print(response['text'])
        
        if found_images:
            print("\n--- Referenced Images ---")
            for img_path in found_images:
                print(f"Image File: {img_path}")
                try:
                    # This opens the image 
                    image_obj = Image.open(img_path)
                    image_obj.show() 
                except Exception as e:
                    print(f"Could not display image: {e}")


# 6. MAIN EXECUTION

if __name__ == "__main__":
    # 1. Setup Text RAG
    txt_store = setup_text_rag()
    
    # 2. Setup Image KG (Uses raw Neo4j Driver)
    neo4j_driver = setup_image_knowledge_graph()
    
    # 3. Start Chat
    try:
        chat_loop(txt_store, neo4j_driver)
    finally:
        # Close the driver when script finishes
        if neo4j_driver:
            neo4j_driver.close()