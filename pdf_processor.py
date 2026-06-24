"""
PDF Document Processor
Handles text extraction and page-aware document chunking for the RAG pipeline.
"""

from typing import List, Dict, Any
from pypdf import PdfReader

class PDFProcessor:
    """Parses PDF documents and chunks the content with structural metadata."""
    
    @staticmethod
    def extract_text_by_page(file_like_object) -> List[Dict[str, Any]]:
        """
        Extracts raw text from a PDF file page by page.
        Args:
            file_like_object: File-like object (e.g. uploaded file from Streamlit).
        Returns:
            List[Dict[str, Any]]: List of pages, each with 'page_number' and 'text'.
        """
        reader = PdfReader(file_like_object)
        pages_data = []
        
        for i, page in enumerate(reader.pages):
            page_number = i + 1
            # Extract text (fallback to empty string if extraction fails)
            text = page.extract_text() or ""
            # Clean up whitespace slightly but preserve line endings
            cleaned_text = " ".join(text.split())
            if cleaned_text.strip():
                pages_data.append({
                    "page_number": page_number,
                    "text": cleaned_text
                })
        return pages_data

    @staticmethod
    def chunk_pages(pages_data: List[Dict[str, Any]], 
                    chunk_size: int = 800, 
                    chunk_overlap: int = 150) -> List[Dict[str, Any]]:
        """
        Splits page text into overlapping, word-bounded chunks while retaining page context.
        Args:
            pages_data (List[Dict[str, Any]]): List of extracted page dicts.
            chunk_size (int): Max character length of each chunk.
            chunk_overlap (int): Overlap character length between sequential chunks.
        Returns:
            List[Dict[str, Any]]: Chunks of text with metadata. Format:
                                  [{"text": "...", "metadata": {"page_number": X}}]
        """
        chunks = []
        
        for page in pages_data:
            page_num = page["page_number"]
            text = page["text"]
            words = text.split()
            
            if not words:
                continue
                
            current_words = []
            current_len = 0
            
            # Reconstruct chunks while maintaining word boundaries
            for word in words:
                current_words.append(word)
                current_len += len(word) + 1  # count word + space
                
                if current_len >= chunk_size:
                    # Save completed chunk
                    chunks.append({
                        "text": " ".join(current_words),
                        "metadata": {"page_number": page_num}
                    })
                    
                    # Compute overlap (estimate number of words based on avg word length ~5.5 chars)
                    approx_overlap_words = max(1, int(chunk_overlap / 6.0))
                    # Retain overlap words
                    current_words = current_words[-approx_overlap_words:] if len(current_words) > approx_overlap_words else []
                    current_len = sum(len(w) + 1 for w in current_words)
            
            # Store residual words
            if current_words:
                chunks.append({
                    "text": " ".join(current_words),
                    "metadata": {"page_number": page_num}
                })
                
        return chunks
