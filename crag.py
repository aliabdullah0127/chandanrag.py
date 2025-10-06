import os
import csv
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
app = FastAPI(title="Enhanced PDF RAG Health Assistant")

# -----------------------------
# Configuration
# -----------------------------
class Config:
    CSV_DIR = "embeddings_data"
    EMBEDDINGS_CSV = os.path.join(CSV_DIR, "embeddings.csv")
    METADATA_CSV = os.path.join(CSV_DIR, "metadata.csv")
    PDF_STORAGE_DIR = "pdf_storage"
    CHUNK_SIZE = 300  # Increased for better context
    CHUNK_OVERLAP = 50  # Add overlap between chunks
    MAX_RETRIEVAL_DOCS = 5  # Increased for better context

# Create necessary directories
os.makedirs(Config.CSV_DIR, exist_ok=True)
os.makedirs(Config.PDF_STORAGE_DIR, exist_ok=True)

# -----------------------------
# Azure OpenAI Setup
# -----------------------------
embeddings_client = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

chat_client = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.1  # Lower temperature for more consistent responses
)

# -----------------------------
# CSV-based Vector Database Class
# -----------------------------
class CSVVectorDB:
    def __init__(self):
        self.embeddings_file = Config.EMBEDDINGS_CSV
        self.metadata_file = Config.METADATA_CSV
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV files if they don't exist"""
        if not os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['chunk_id', 'embedding'])
        
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['chunk_id', 'text', 'source', 'page_number', 'chunk_index', 'timestamp'])
    
    def _generate_chunk_id(self, text: str, source: str, chunk_index: int) -> str:
        """Generate unique ID for a chunk"""
        content = f"{text}_{source}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def add_texts(self, texts: List[str], source: str, page_numbers: List[int] = None) -> List[str]:
        """Add texts and their embeddings to CSV storage"""
        chunk_ids = []
        timestamp = datetime.now().isoformat()
        
        if page_numbers is None:
            page_numbers = [0] * len(texts)
        
        try:
            # Generate embeddings for all texts at once (more efficient)
            embeddings_vectors = await embeddings_client.aembed_documents(texts)
            
            # Prepare data for batch writing
            embedding_rows = []
            metadata_rows = []
            
            for i, (text, embedding_vector) in enumerate(zip(texts, embeddings_vectors)):
                chunk_id = self._generate_chunk_id(text, source, i)
                chunk_ids.append(chunk_id)
                
                # Convert embedding to JSON string for CSV storage
                embedding_json = json.dumps(embedding_vector)
                embedding_rows.append([chunk_id, embedding_json])
                
                page_num = page_numbers[i] if i < len(page_numbers) else 0
                metadata_rows.append([chunk_id, text, source, page_num, i, timestamp])
            
            # Write embeddings to CSV
            with open(self.embeddings_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(embedding_rows)
            
            # Write metadata to CSV
            with open(self.metadata_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
            
            logger.info(f"Successfully added {len(texts)} chunks from {source}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding texts to CSV DB: {str(e)}")
            raise
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        try:
            # Generate embedding for the query
            query_embedding = await embeddings_client.aembed_query(query)
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            # Load embeddings from CSV
            embeddings_df = pd.read_csv(self.embeddings_file)
            metadata_df = pd.read_csv(self.metadata_file)
            
            if embeddings_df.empty:
                return []
            
            # Calculate similarities
            similarities = []
            for _, row in embeddings_df.iterrows():
                embedding_vector = np.array(json.loads(row['embedding'])).reshape(1, -1)
                similarity = cosine_similarity(query_vector, embedding_vector)[0][0]
                similarities.append((row['chunk_id'], similarity))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunk_ids = [chunk_id for chunk_id, _ in similarities[:k]]
            
            # Get metadata for top chunks
            results = []
            for chunk_id in top_chunk_ids:
                metadata_row = metadata_df[metadata_df['chunk_id'] == chunk_id]
                if not metadata_row.empty:
                    row = metadata_row.iloc[0]
                    results.append({
                        'chunk_id': chunk_id,
                        'text': row['text'],
                        'source': row['source'],
                        'page_number': row['page_number'],
                        'similarity': next(sim for cid, sim in similarities if cid == chunk_id)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def get_document_stats(self) -> Dict:
        """Get statistics about stored documents"""
        try:
            metadata_df = pd.read_csv(self.metadata_file)
            if metadata_df.empty:
                return {"total_chunks": 0, "documents": []}
            
            stats = {
                "total_chunks": len(metadata_df),
                "documents": metadata_df['source'].value_counts().to_dict(),
                "latest_upload": metadata_df['timestamp'].max() if 'timestamp' in metadata_df.columns else None
            }
            return stats
        except Exception:
            return {"total_chunks": 0, "documents": []}

# Initialize the CSV Vector Database
vector_db = CSVVectorDB()

# -----------------------------
# Enhanced PDF Processing
# -----------------------------
def extract_text_with_page_numbers(file_path: str) -> List[Tuple[str, int]]:
    """Extract text from PDF with page numbers"""
    try:
        reader = PdfReader(file_path)
        text_with_pages = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                text_with_pages.append((text.strip(), page_num))
        
        return text_with_pages
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return []

def create_overlapping_chunks(text: str, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
    """Create overlapping text chunks for better context preservation"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 10:  # Only keep substantial chunks
            chunks.append(chunk.strip())
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Enhanced PDF upload with better processing"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        file_path = os.path.join(Config.PDF_STORAGE_DIR, file.filename)
        contents = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Extract text with page numbers
        text_with_pages = extract_text_with_page_numbers(file_path)
        
        if not text_with_pages:
            return JSONResponse(
                status_code=400,
                content={"error": "No readable text found in PDF"}
            )
        
        # Process chunks with page information
        all_chunks = []
        all_page_numbers = []
        
        for text, page_num in text_with_pages:
            chunks = create_overlapping_chunks(text)
            all_chunks.extend(chunks)
            all_page_numbers.extend([page_num] * len(chunks))
        
        # Add to vector database
        chunk_ids = await vector_db.add_texts(all_chunks, file.filename, all_page_numbers)
        
        return {
            "filename": file.filename,
            "chunks_stored": len(chunk_ids),
            "pages_processed": len(text_with_pages),
            "message": "PDF uploaded and processed successfully"
        }
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask")
async def ask_question(request: dict):
    """Enhanced question answering with better context handling"""
    try:
        question = request.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Search for relevant documents
        relevant_docs = await vector_db.similarity_search(question, k=Config.MAX_RETRIEVAL_DOCS)
        
        if not relevant_docs:
            return {
                "response": "I'm sorry, I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": []
            }
        
        # Prepare context with source information
        context_parts = []
        sources = []
        
        for doc in relevant_docs:
            context_parts.append(f"[Source: {doc['source']}, Page: {doc['page_number']}]\n{doc['text']}")
            sources.append({
                "source": doc['source'],
                "page": doc['page_number'],
                "similarity": round(doc['similarity'], 3)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Enhanced system prompt
        system_prompt = """You are a knowledgeable and helpful medical assistant. 
        
        Guidelines:
        1. Use ONLY the information provided in the context below
        2. Be accurate and cite specific sources when possible
        3. If the information is not in the context, clearly state so
        4. Maintain a professional yet friendly tone
        5. If medical advice is requested, remind users to consult healthcare professionals
        
        Always structure your response clearly and mention relevant sources."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]
        
        # Generate response
        response = await chat_client.agenerate(messages=[messages])
        answer = response.generations[0][0].text
        
        return {
            "response": answer,
            "sources": sources,
            "context_chunks_used": len(relevant_docs)
        }
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        stats = vector_db.get_document_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.delete("/clear-database")
async def clear_database():
    """Clear all stored embeddings and metadata"""
    try:
        # Remove CSV files
        if os.path.exists(Config.EMBEDDINGS_CSV):
            os.remove(Config.EMBEDDINGS_CSV)
        if os.path.exists(Config.METADATA_CSV):
            os.remove(Config.METADATA_CSV)
        
        # Reinitialize
        vector_db._initialize_csv_files()
        
        return {"message": "Database cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear database")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
