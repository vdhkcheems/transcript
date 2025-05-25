import sys
import argparse
from utils import TranscriptProcessor, TFIDFSearcher, SentenceTransformerSearcher


def main():
    if len(sys.argv) != 3:
        print("Usage: python transcript.py <transcript_file> <method>")
        print("Methods: tfidf, sentence_transformer")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    method = sys.argv[2]
    
    # Validate method
    if method not in ['tfidf', 'llm1']:
        print("Error: Method must be 'tfidf' or 'llm1'")
        sys.exit(1)
    
    try:
        # Load and process transcript
        processor = TranscriptProcessor()
        chunks = processor.load_and_chunk_transcript(transcript_file)
        
        # Initialize the appropriate searcher
        if method == 'tfidf':
            searcher = TFIDFSearcher(chunks)
        elif method == 'llm1':
            searcher = SentenceTransformerSearcher(chunks)
        
        print("Transcript loaded, please ask your question (press 8 for exit):")
        
        # Interactive question-answering loop
        while True:
            question = input("\nQuestion: ").strip()
            
            # Exit condition
            if question == '8':
                print("Goodbye!")
                break
            
            if not question:
                print("Please enter a valid question or '8' to exit.")
                continue
            
            try:
                # Get the best matching chunk
                best_chunk, score = searcher.search(question)
                
                if best_chunk:
                    # Format and display result
                    timestamp_range = processor.get_timestamp_range(best_chunk)
                    print(f"\n[{timestamp_range}], {best_chunk['text']}")
                    print(f"(Confidence: {score:.3f})")
                else:
                    print("No relevant information found.")
                    
            except Exception as e:
                print(f"Error processing question: {e}")
                
    except FileNotFoundError:
        print(f"Error: File '{transcript_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()