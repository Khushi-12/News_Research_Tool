# News Research Tool

This project is a News Research Tool that allows users to input URLs of news articles, process them to create embeddings, and answer questions based on the content of these articles. The tool uses a pre-trained question-answering model that can be fine-tuned for better performance.

## Features
- **URL Processing:** Input up to 3 URLs of news articles.
- **Text Splitting:** Split the content of the articles into manageable chunks for processing.
- **Embeddings:** Create embeddings from the text chunks using a pre-trained model.
- **Question Answering:** Input a query and get answers based on the content of the articles.

## Technologies Used
- **Streamlit:** For the web interface.
- **Hugging Face Transformers:** For pre-trained models and tokenization.
- **LangChain:** For document loading, splitting, and vector storage.
- **FAISS:** For efficient similarity search.

## Getting Started

### Prerequisites
- Python 3.7+
- `pip` (Python package installer)

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/news-research-tool.git
    cd news-research-tool
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Directory Structure
news-research-tool/
│
├── model.py # Contains the QuestionAnsweringAgent class
├── app.py # Main Streamlit application file
├── requirements.txt # Required Python packages
└── README.md # This readme file

### Usage

1. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2. **Input URLs:**
    - In the sidebar, input up to 3 URLs of news articles.
    - Click the "Process URL" button to load and process the data.

3. **Ask a Question:**
    - Enter your query in the text input field.
    - The application will generate an answer based on the content of the articles.
