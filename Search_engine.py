import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import PyPDF2
import markdown
from dataclasses import dataclass
import re
import argparse
from tqdm import tqdm
import torch

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    text: str
    file_path: str
    file_type: str
    chunk_id: int
    page_num: Optional[int] = None
    section_title: Optional[str] = None

class DocumentProcessor:
    """Handles reading and processing different file types"""
    
    @staticmethod
    def read_pdf(file_path: str) -> List[str]:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pages = []
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        pages.append((text, page_num + 1))
                return pages
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return []
    
    @staticmethod
    def read_markdown(file_path: str) -> str:
        """Read markdown file and convert to plain text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Convert markdown to HTML then extract text
                html = markdown.markdown(content)
                # Simple HTML tag removal
                text = re.sub(r'<[^>]+>', '', html)
                return text
        except Exception as e:
            print(f"Error reading Markdown {file_path}: {e}")
            return ""
    
    @staticmethod
    def read_text(file_path: str) -> str:
        """Read plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
        
    @staticmethod
    def read_html(file_path: str) -> str:
        """Read HTML file and extract plain text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Simple HTML tag removal
                text = re.sub(r'<[^>]+>', '', content)
                return text
        except Exception as e:
            print(f"Error reading HTML file {file_path}: {e}")
            return ""

class TextChunker:
    """Handles intelligent text chunking for better search results"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, file_path: str, file_type: str, page_num: Optional[int] = None) -> List[Document]:
        """Split text into overlapping chunks"""
        # Clean text
        text = self._clean_text(text)
        
        # Split by sentences first
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(Document(
                    text=current_chunk.strip(),
                    file_path=file_path,
                    file_type=file_type,
                    chunk_id=chunk_id,
                    page_num=page_num
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk.split())
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Document(
                text=current_chunk.strip(),
                file_path=file_path,
                file_type=file_type,
                chunk_id=chunk_id,
                page_num=page_num
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for next chunk"""
        words = text.split()
        if len(words) <= self.overlap:
            return text
        return " ".join(words[-self.overlap:])

class AcademicSearchEngine:
    """Main search engine class"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto"):
        """Initialize the search engine"""
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load SBERT model
        print("Loading SBERT model...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

        # Initialize components
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.chunker = TextChunker()
        self.processor = DocumentProcessor()
        
        # Cache file paths
        self.cache_dir = Path(".search_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def index_folder(self, folder_path: str, force_rebuild: bool = False) -> None:
        """Index all supported files in a folder"""
        folder_path = Path(folder_path)
        cache_file = self.cache_dir / f"index_{folder_path.name}.pkl"

        print(f"Cache file: {cache_file}")

        # cache_file = r'C:\Users\jonas\code\AdvSpec\.search_cache.pkl'
        
        # Check if cache exists and is recent
        if not force_rebuild:# and cache_file.exists():
            print("Loading cached index...")
            self._load_cache(cache_file)
            return
        
        print(f"Indexing folder: {folder_path}")
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.md', '.txt', '.markdown', '.html'}
        
        # Find all supported files
        files = []
        for ext in supported_extensions:
            files.extend(folder_path.rglob(f"*{ext}"))
        
        print(f"Found {len(files)} files to process")
        
        # Process each file
        all_documents = []
        for file_path in tqdm(files, desc="Processing files"):
            documents = self._process_file(file_path)
            all_documents.extend(documents)
        
        print(f"Created {len(all_documents)} document chunks")
        
        # Store documents
        self.documents = all_documents
        
        # Create embeddings
        print("Creating embeddings...")
        texts = [doc.text for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS index
        print("Building search index...")
        self._build_faiss_index()
        
        # Save cache
        self._save_cache(cache_file)
        print("Indexing complete!")
    
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single file and return document chunks"""
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.pdf':
                pages = self.processor.read_pdf(str(file_path))
                documents = []
                for text, page_num in pages:
                    chunks = self.chunker.chunk_text(text, str(file_path), 'pdf', page_num)
                    documents.extend(chunks)
                return documents
            
            elif file_ext in ['.md', '.markdown']:
                text = self.processor.read_markdown(str(file_path))
                return self.chunker.chunk_text(text, str(file_path), 'markdown')
            
            elif file_ext == '.txt':
                text = self.processor.read_text(str(file_path))
                return self.chunker.chunk_text(text, str(file_path), 'text')
            
            elif file_ext == '.html':
                text = self.processor.read_html(str(file_path))
                return self.chunker.chunk_text(text, str(file_path), 'html')
   
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return []
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for fast similarity search"""
        if self.embeddings is None:
            raise ValueError("No embeddings available")
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype(np.float32))
    
    def search(self, query: str, top_k: int = 20, min_score: float = 0.2) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        if self.index is None or self.embeddings is None:
            raise ValueError("Index not built. Call index_folder() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), 1000)


        candidates = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

        # print(candidates)

        scores_ce = self.cross_encoder.predict([(query, str(doc)) for doc in candidates])

        # print(scores_ce)

        scores = [scores_ce+10]
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores_ce, indices[0])):
            if score >= min_score and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _save_cache(self, cache_file: Path) -> None:
        """Save index to cache"""
        cache_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Save FAISS index separately
        if self.index is not None:
            faiss.write_index(self.index, str(cache_file.with_suffix('.faiss')))
    
    def _load_cache(self, cache_file: Path) -> None:
        """Load index from cache"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.documents = cache_data['documents']
        self.embeddings = cache_data['embeddings']
        
        # Load FAISS index
        faiss_file = cache_file.with_suffix('.faiss')
        if faiss_file.exists():
            self.index = faiss.read_index(str(faiss_file))


    def print_results(self, results: List[Tuple[Document, float]], max_text_length: int = 600, output_html: str = "search_results.html") -> None:
        """Print search results in a readable format and generate an HTML file"""
        if not results:
            print("No results found.")
            return
        
        # Start HTML content
        html_content = "<html><body><h1>Search Results</h1>"
        
        for i, (doc, score) in enumerate(results, 1):
            html_content += f"<h2>Result {i} (Score: {score:.3f})</h2>"
            html_content += f"<p><strong>File:</strong> {doc.file_path}</p>"
            
            file_path = doc.file_path.replace('\\', '/')
            # Create a clickable link to the specific page or section
            if doc.page_num:
                link = f"file:///{file_path}"  # Replace backslashes with forward slashes
                link += f"#page={doc.page_num}"  # Append page number
            elif doc.section_title:
                print(f"Link: {file_path}#section={doc.section_title}")  # Section link
            else:
                link = f"file:///{file_path}"  # Replace backslashes with forward slashes
            
            html_content += f'<p><strong>Link:</strong> <a href="{link}" target="_blank">{link}</a></p>'
            
            if doc.page_num:
                print(f"Page: {doc.page_num}")
                html_content += f"<p><strong>Page:</strong> {doc.page_num}</p>"
            html_content += f"<p><strong>Type:</strong> {doc.file_type}</p>"
            
            print(f"Type: {doc.file_type}")
            print(f"Link: {link}")
            
            # Truncate text if too long
            text = doc.text
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            print(f"Content: {text}")
            print("-" * 50)

            html_content += f"<p><strong>Content:</strong> {text}</p>"
            html_content += "<hr>"
        
        # End HTML content
        html_content += "</body></html>"
        
        # Save HTML file
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Results saved to {output_html}. Open this file in a browser to view clickable links.")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Academic Search Engine")
    parser.add_argument("--folder", required=True, help="Folder path to index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SBERT model name")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Initialize search engine
    engine = AcademicSearchEngine(model_name=args.model, device=args.device)
    
    # Index folder
    engine.index_folder(args.folder, force_rebuild=args.rebuild)
    
    # Interactive search loop
    print("\n=== Academic Search Engine ===")
    print("Enter your exam questions or queries. Type 'quit' to exit.")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # try:
        results = engine.search(query, top_k=20)
        engine.print_results(results)
        # except Exception as e:
            # print(f"Search error: {e}")

if __name__ == "__main__":
    # If running directly, start CLI
    main()