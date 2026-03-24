import os
from dotenv import load_dotenv
from PIL import Image

# LangChain & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

# Neo4j Driver
from neo4j import GraphDatabase

# 1. Load Environment Variables
load_dotenv()

# CONFIGURATION
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MODEL_PATH = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class RAGQueryEngine:
    def __init__(self):
        print("--- Initializing Query Engine ---")
        
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Initialize LLM
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_tokens=512,
            n_ctx=2048,
            verbose=False
        )

        # Connect to Pinecone
        self.vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=self.embeddings
        )

        # Connect to Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def get_response(self, query: str):
        # 1. Retrieve Text Context from Pinecone
        docs = self.vectorstore.similarity_search(query, k=3)
        context_text = "\n".join([d.page_content for d in docs])

        # 2. Retrieve Image Context from Neo4j
        query_embedding = self.embeddings.embed_query(query)
        image_search_query = """
        CALL db.index.vector.queryNodes('photo_desc_index', 3, $embedding)
        YIELD node, score
        WHERE score > 0.4
        RETURN node.path AS path, node.description AS description, score
        """
        
        found_images = []
        image_context_str = ""
        
        with self.neo4j_driver.session() as session:
            result = session.run(image_search_query, embedding=query_embedding)
            records = list(result)
            if records:
                found_images = [record["path"] for record in records]
                descriptions = [record["description"] for record in records]
                image_context_str = f"\nRelevant Image Descriptions: {', '.join(descriptions)}"

        # 3. Generate LLM Response
        final_context = f"PDF Context:\n{context_text}\n{image_context_str}"
        
        template = """You are a helpful assistant. Use the following context to answer the question.
        Context: {context}
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        
        response = llm_chain.invoke({"context": final_context, "question": query})
        return response['text'], found_images

    def close(self):
        self.neo4j_driver.close()

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    engine = RAGQueryEngine()
    
    print("\n" + "="*50)
    print("QUERY SYSTEM READY. (Type 'exit' to quit)")
    print("="*50 + "\n")

    try:
        while True:
            user_query = input("User Query: ")
            if user_query.lower() in ['exit', 'quit']:
                break

            answer, images = engine.get_response(user_query)
            
            print("\n--- AI Response ---")
            print(answer)
            
            if images:
                print("\n--- Referenced Images ---")
                for img_path in images:
                    print(f"Opening: {img_path}")
                    try:
                        img = Image.open(img_path)
                        img.show()
                    except:
                        print(f"Could not open image file at {img_path}")
            print("\n" + "-"*30 + "\n")

    finally:
        engine.close()