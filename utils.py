import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TranscriptProcessor:
    """Handles transcript loading, parsing, and chunking"""
    
    def __init__(self, chunk_target_size=300):
        """
        Initialize processor
        Args:
            chunk_target_size: Target number of words per chunk
        """
        self.chunk_target_size = chunk_target_size
    
    def load_and_chunk_transcript(self, file_path):
        """
        Load transcript from file and create semantic chunks
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            List of chunk dictionaries with text and timestamp metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Parse transcript entries
        entries = self._parse_transcript(content)
        
        # Create semantic chunks
        chunks = self._create_chunks(entries)
        
        print(f"Loaded {len(entries)} transcript entries, created {len(chunks)} chunks")
        return chunks
    
    def _parse_transcript(self, content):
        """
        Parse transcript content into structured entries
        
        Args:
            content: Raw transcript text
            
        Returns:
            List of dictionaries with timestamp and text
        """
        entries = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract timestamp and text using regex
            match = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if match:
                timestamp = match.group(1)
                text = match.group(2).strip()
                if text:  # Only add non-empty text
                    entries.append({
                        'timestamp': timestamp,
                        'text': text
                    })
        
        return entries
    
    def _create_chunks(self, entries):
        """
        Create semantic chunks from transcript entries
        
        Args:
            entries: List of parsed transcript entries
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk_entries = []
        current_word_count = 0
        
        for entry in entries:
            word_count = len(entry['text'].split())
            current_chunk_entries.append(entry)
            current_word_count += word_count
            
            # Create chunk when we reach target size or at natural breaks
            should_chunk = (
                current_word_count >= self.chunk_target_size or
                self._is_natural_break(entry['text'])
            )
            
            if should_chunk and current_chunk_entries:
                chunk = self._build_chunk(current_chunk_entries)
                chunks.append(chunk)
                current_chunk_entries = []
                current_word_count = 0
        
        # Add remaining entries as final chunk
        if current_chunk_entries:
            chunk = self._build_chunk(current_chunk_entries)
            chunks.append(chunk)
        
        return chunks
    
    def _is_natural_break(self, text):
        """
        Check if text indicates a natural breaking point
        
        Args:
            text: Text to analyze
            
        Returns:
            Boolean indicating if this is a good break point
        """
        break_indicators = [
            'so let\'s', 'now let\'s', 'moving on', 'next topic',
            'in conclusion', 'to summarize', 'finally',
            'first', 'second', 'third', 'lastly'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in break_indicators)
    
    def _build_chunk(self, entries):
        """
        Build a chunk dictionary from a list of entries
        
        Args:
            entries: List of transcript entries
            
        Returns:
            Chunk dictionary with text and timestamp metadata
        """
        combined_text = ' '.join(entry['text'] for entry in entries)
        timestamps = {}
        
        current_pos = 0
        for entry in entries:
            text = entry['text']
            start_pos = current_pos
            end_pos = current_pos + len(text)
            timestamps[entry['timestamp']] = [start_pos, end_pos]
            current_pos = end_pos + 1  # +1 for space separator
        
        return {
            'text': combined_text,
            'timestamps': timestamps
        }
    
    def get_timestamp_range(self, chunk):
        """
        Get the timestamp range for a chunk
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            String representing the timestamp range
        """
        if not chunk['timestamps']:
            return "Unknown"
        
        timestamps = list(chunk['timestamps'].keys())
        if len(timestamps) == 1:
            return timestamps[0]
        
        # Extract start and end times
        first_timestamp = timestamps[0]
        last_timestamp = timestamps[-1]
        
        # Parse start time from first timestamp
        start_time = first_timestamp.split(' - ')[0]
        # Parse end time from last timestamp  
        end_time = last_timestamp.split(' - ')[1]
        
        return f"{start_time} - {end_time}"


class TFIDFSearcher:
    """TF-IDF based semantic search"""
    
    def __init__(self, chunks):
        """
        Initialize TF-IDF searcher
        
        Args:
            chunks: List of text chunks
        """
        self.chunks = chunks
        self.texts = [chunk['text'] for chunk in chunks]
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_features=5000,
            lowercase=True
        )
        
        print("Building TF-IDF vectors...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        print("TF-IDF initialization complete!")
    
    def search(self, question):
        """
        Search for the most relevant chunk using TF-IDF
        
        Args:
            question: User question
            
        Returns:
            Tuple of (best_chunk, similarity_score)
        """
        # Vectorize the question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(question_vector, self.tfidf_matrix)
        similarities = similarities.flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > 0:
            return self.chunks[best_idx], best_score
        else:
            return None, 0


class SentenceTransformerSearcher:
    """Sentence transformer based semantic search"""
    
    def __init__(self, chunks, model_name='BAAI/bge-base-en-v1.5'):
        """
        Initialize sentence transformer searcher
        
        Args:
            chunks: List of text chunks
            model_name: Name of the sentence transformer model
        """
        self.chunks = chunks
        self.texts = [chunk['text'] for chunk in chunks]
        
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print("Encoding text chunks...")
        self.chunk_embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        print("Sentence transformer initialization complete!")
    
    def search(self, question):
        """
        Search for the most relevant chunk using sentence transformers
        
        Args:
            question: User question
            
        Returns:
            Tuple of (best_chunk, similarity_score)
        """
        # Encode the question
        question_embedding = self.model.encode([question], convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(
            question_embedding.cpu().numpy(),
            self.chunk_embeddings.cpu().numpy()
        )
        similarities = similarities.flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return self.chunks[best_idx], best_score