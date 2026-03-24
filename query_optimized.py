import os
import concurrent.futures  
from dotenv import load_dotenv
from PIL import Image

# LangChain imports
import langchain
from langchain_classic.globals import set_llm_cache
from langchain_community.cache import InMemoryCache 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from neo4j import GraphDatabase

load_dotenv()

# 1. ENABLE CACHING
# InMemoryCache resets when you restart. 
# Use 'langchain.cache.SQLiteCache(database_path=".langchain.db")' for a permanent cache.
set_llm_cache(InMemoryCache())

class RAGQueryEngine:
    def __init__(self):
        print("--- Initializing Optimized Query Engine (CPU) ---")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. OPTIMIZE FOR CPU
        self.llm = LlamaCpp(
            model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.1,
            max_tokens=512,
            n_ctx=2048,
            n_threads=6,      
            n_batch=512,      
            verbose=False
        )

        self.vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"), 
            embedding=self.embeddings
        )

        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    # Helper function for Pinecone retrieval
    def _get_pinecone_data(self, query):
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])

    # Helper function for Neo4j retrieval
    def _get_neo4j_data(self, query_embedding):
        image_search_query = """
        CALL db.index.vector.queryNodes('photo_desc_index', 3, $embedding)
        YIELD node, score
        WHERE score > 0.4
        RETURN node.path AS path, node.description AS description
        """
        found_images = []
        image_descriptions = []
        
        with self.neo4j_driver.session() as session:
            result = session.run(image_search_query, embedding=query_embedding)
            for record in result:
                found_images.append(record["path"])
                image_descriptions.append(record["description"])
        
        desc_str = f"\nRelevant Image Descriptions: {', '.join(image_descriptions)}" if image_descriptions else ""
        return found_images, desc_str

    def get_response(self, query: str):
        # Generate embedding once (used for Neo4j search)
        query_embedding = self.embeddings.embed_query(query)

        # 3. PARALLELIZATION: Fetch from Pinecone and Neo4j simultaneously
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_pinecone = executor.submit(self._get_pinecone_data, query)
            future_neo4j = executor.submit(self._get_neo4j_data, query_embedding)

            # Collect results as they finish
            context_text = future_pinecone.result()
            found_images, image_context_str = future_neo4j.result()

        # Generate final response
        final_context = f"PDF Context:\n{context_text}\n{image_context_str}"
        
        template = """Use the context only below to answer the question. Do not make up information that is not in the context and use bullet points and emojis where necessary.\nContext: {context}\nQuestion: {question}\nAnswer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        
        print("\n--- AI Response ---")
        response = llm_chain.invoke({"context": final_context, "question": query})
        return response['text'], found_images

    def close(self):
        self.neo4j_driver.close()

if __name__ == "__main__":
    engine = RAGQueryEngine()
    try:
        while True:
            user_query = input("User Query: ")
            if user_query.lower() in ['exit', 'quit']: break

            answer, images = engine.get_response(user_query)
            print(answer)
            
            if images:
                for img_path in images:
                    try:
                        Image.open(img_path).show()
                    except: pass
            print("-" * 30)
    finally:
        engine.close()