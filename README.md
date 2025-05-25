# Semantic Search Transcript Application

A Python-based application that performs semantic search on transcript files using TF-IDF vectorization and Sentence Transformers.

## Features

- **TF-IDF Search**: Uses Term Frequency-Inverse Document Frequency with cosine similarity
- **Sentence Transformer Search**: Uses pre-trained neural models for semantic understanding
- **Smart Chunking**: Combines transcript segments into meaningful chunks while preserving timestamp metadata
- **Interactive CLI**: Command-line interface for real-time question answering

## Installation

1. Clone or download the project files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the application using one of these commands:

```bash
# Using TF-IDF method
python transcript.py transcript.txt tfidf

# Using Sentence Transformer method  
python transcript.py transcript.txt llm1
```

### Interactive Session

After running the command, you'll see:
```
Transcript loaded, please ask your question (press 8 for exit):
```

- Type your question and press Enter
- The system will return the most relevant text chunk with timestamp
- Type `8` to exit the program

### Example Questions

Try asking questions like:
- "What is artificial intelligence?"
- "When was AI first coined?"
- "How are ML and AI related?"
- "Why is AI popular now?"

## Project Structure

```
├── transcript.py          # Main application file
├── utils.py              # Core processing classes
├── transcript.txt        # Sample transcript file
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## How It Works

### 1. Transcript Processing
- Parses transcript files with timestamp format: `[HH:MM - HH:MM] text`
- Creates intelligent chunks that combine multiple timestamp entries
- Preserves timestamp metadata for each chunk

### 2. Search Methods

#### TF-IDF Method
- Converts text to numerical vectors using term frequency analysis
- Calculates cosine similarity between question and chunks
- Returns chunk with highest similarity score

#### Sentence Transformer Method (llm1)
- Uses pre-trained neural models (BAAI/bge-base-en-v1.5)
- Creates dense vector embeddings that capture semantic meaning
- More accurate for understanding context and meaning

### 3. Chunking Strategy
- Combines transcript entries into ~300-word chunks
- Maintains timestamp boundaries (chunks never split timestamps)
- Looks for natural break points in content
- Preserves character-level mapping of timestamps within chunks

## Technical Details

### Chunk Structure
Each chunk contains:
```python
{
    "text": "combined text from multiple timestamps",
    "timestamps": {
        "00:00 - 00:10": [0, 65],    # character positions
        "00:10 - 00:12": [65, 98],   # in the chunk text
    }
}
```

### Dependencies
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **sentence-transformers**: Pre-trained semantic models
- **numpy**: Numerical computations
- **torch**: Backend for sentence transformers
- **transformers**: Hugging Face model support

## Performance Notes

- **TF-IDF**: Fast initialization, good for keyword-based queries
- **Sentence Transformers**: Slower initialization but better semantic understanding
- First run downloads the sentence transformer model

## Customization

### Modify Chunk Size
In `utils.py`, adjust the `chunk_target_size` parameter:
```python
processor = TranscriptProcessor(chunk_target_size=500)  # Larger chunks
```

### Change Sentence Transformer Model
In `utils.py`, modify the `model_name` parameter:
```python
searcher = SentenceTransformerSearcher(chunks, model_name='all-mpnet-base-v2')
```

## Troubleshooting

### Common Issues
1. **Module not found**: Run `pip install -r requirements.txt`
2. **Slow first run**: Sentence transformer downloads model on first use
3. **No results**: Try rephrasing questions or check transcript format

### Transcript Format
Ensure your transcript follows this format:
```
[HH:MM - HH:MM]  Your text here
[HH:MM - HH:MM]  More text here
```