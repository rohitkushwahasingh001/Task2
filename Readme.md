# RAG-Based Semantic Quote Retrieval and Structured QA System

This project implements a Retrieval Augmented Generation (RAG) pipeline for semantically retrieving quotes and providing structured answers and explanations based on natural language queries. It leverages fine-tuned sentence embeddings, FAISS for efficient search, and a Large Language Model (LLM) for generative capabilities, all wrapped in an interactive Streamlit web application.

## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
3.  [Technologies Used](#technologies-used)
4.  [Setup Instructions](#setup-instructions)
    * [Prerequisites](#prerequisites)
    * [Clone the Repository](#clone-the-repository)
    * [Create Virtual Environment](#create-virtual-environment)
    * [Install Dependencies](#install-dependencies)
    * [Set Up Groq API Key](#set-up-groq-api-key)
5.  [Running the Project](#running-the-project)
    * [Step 1: Preprocess Data](#step-1-preprocess-data)
    * [Step 2: Fine-Tune the Embedding Model](#step-2-fine-tune-the-embedding-model)
    * [Step 3: Build the FAISS Vector Store](#step-3-build-the-faiss-vector-store)
    * [Step 4: Run the Streamlit Application](#step-4-run-the-streamlit-application)
    * [Step 5: Run RAG Evaluation (Optional)](#step-5-run-rag-evaluation-optional)
6.  [Example Queries](#example-queries)
7.  [Evaluation](#evaluation)
8.  [Future Enhancements](#future-enhancements)
9.  [License](#license)

---

## Features

* **Semantic Quote Retrieval**: Find quotes based on their meaning, not just keywords.
* **Fine-tuned Embeddings**: Uses a `SentenceTransformer` model fine-tuned on quotes for improved relevance.
* **Efficient Search**: Leverages FAISS for fast nearest-neighbor search in the vector space.
* **LLM-Powered Explanations**: Integrates with Groq's Llama3 LLM to provide insightful explanations for retrieved quotes.
* **Interactive Web App**: User-friendly interface built with Streamlit for easy interaction.
* **Modular Design**: Clear separation of concerns with dedicated scripts for data preprocessing, model training, index building, and application logic.

## Project Structure

```
task2_rag_quote_retrieval/
│
├── .env                  # Stores GROQ_API_KEY
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── preprocess_data.py    # Downloads and cleans dataset
├── train_model.py        # Fine-tunes sentence-transformer model
├── build_rag_pipeline.py # Builds FAISS index
├── app.py                # Streamlit chat-style web app
├── evaluate_rag.py       # Script for RAG evaluation (with dummy data)
│
├── data/                 # Folder to store quotes dataset
│   └── english_quotes.csv
│
├── models/               # Folder to save trained model
│   └── quote_bert_model/
│       ├── (model files like config.json, pytorch_model.bin, etc.)
│
└── vectorstore/          # FAISS index storage
    └── faiss_index.bin
```

## Technologies Used

* **Python 3.8+**
* **Hugging Face `datasets`**: For dataset management.
* **Pandas**: For data manipulation.
* **Sentence Transformers**: For creating and fine-tuning embeddings.
* **FAISS (Facebook AI Similarity Search)**: For vector indexing and search.
* **Groq API**: For Large Language Model (LLM) inference (Llama3-8b-8192).
* **Streamlit**: For the web application interface.
* **Ragas**: For RAG system evaluation.
* **`python-dotenv`**: For environment variable management.
* **`nest_asyncio`**: (Specifically for macOS users) to resolve potential async issues with Streamlit.

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.8 or higher installed.
* `pip` (Python package installer).

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <repository_url>
cd task2_rag_quote_retrieval
```

*(Replace `<repository_url>` with the actual URL of your repository if applicable, otherwise ensure you have all files in the `task2_rag_quote_retrieval` directory.)*

### Create Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

```bash
python -m venv venv
```

**Activate the virtual environment:**

* **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
* **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### Install Dependencies

With your virtual environment activated, install all the required Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
streamlit
sentence-transformers
faiss-cpu
pandas
groq
python-dotenv
datasets
ragas
numpy
nest_asyncio # Required for macOS patch in app.py
```

### Set Up Groq API Key

The `app.py` uses the Groq API for its LLM capabilities. You need to obtain an API key and configure it:

1.  **Obtain API Key**: Go to the [Groq website](https://groq.com/) and sign up to get your API key.
2.  **Create `.env` File**: In the root directory of your project (`task2_rag_quote_retrieval/`), create a new file named `.env` (make sure it's exactly `.env` with the leading dot).
3.  **Add API Key**: Open the `.env` file and add your Groq API key in the following format:

    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

    **Replace `"your_groq_api_key_here"` with your actual API key.**

## Running the Project

Once the setup is complete, you need to run the scripts in a specific order to prepare the data, train the model, build the index, and finally launch the application. Ensure your virtual environment is activated for all these steps.

### Step 1: Preprocess Data

This script downloads and cleans the `Abirate/english_quotes` dataset.

```bash
python preprocess_data.py
```

* **Output**: Creates the `data/english_quotes.csv` file.

### Step 2: Fine-Tune the Embedding Model

This script fine-tunes a `SentenceTransformer` model on the cleaned quote data. This step can take some time depending on your CPU.

```bash
python train_model.py
```

* **Output**: Creates the `models/quote_bert_model/` directory containing the fine-tuned model files.

### Step 3: Build the FAISS Vector Store

This script generates embeddings for all quotes using the fine-tuned model and builds a FAISS index for fast similarity search.

```bash
python build_rag_pipeline.py
```

* **Output**: Creates the `vectorstore/faiss_index.bin` file.

### Step 4: Run the Streamlit Application

Finally, launch the interactive web application.

```bash
streamlit run app.py
```

* This command will open the "💬 RAG Quote Assistant" in your default web browser (usually at `http://localhost:8501`).
* You can now enter natural language queries in the text box and interact with the system.

### Step 5: Run RAG Evaluation (Optional)

This script demonstrates how to use the Ragas framework for RAG evaluation.

```bash
python evaluate_rag.py
```

* **Note**: The current `evaluate_rag.py` uses a *dummy dataset* for demonstration purposes. For a comprehensive evaluation, you would need to integrate it with your actual RAG pipeline to collect real LLM answers, retrieved contexts, and provide accurate ground truth answers.

## Example Queries

Try these queries in the Streamlit app:

* `Quotes about courage by women authors`
* `Motivational quotes tagged 'accomplishment'`
* `All Oscar Wilde quotes with humor`
* `Quotes about insanity attributed to Einstein`
* `Quotes on love and life`

## Evaluation

The `evaluate_rag.py` script is set up to use the `ragas` library for RAG evaluation. Ragas provides metrics such as:

* **Faithfulness**: Measures how factually consistent the generated answer is with the retrieved context.
* **Answer Relevance**: Assesses how relevant the generated answer is to the given question.
* **Context Relevance**: Evaluates if the retrieved contexts are relevant to the question.
* **Context Recall**: Measures if all relevant parts of the ground truth answer are covered by the retrieved context.

*(As noted, the current implementation uses dummy data for demonstration; a full evaluation requires real data collection from the RAG pipeline.)*

## Future Enhancements

* **Comprehensive RAG Evaluation**: Implement a robust data collection mechanism for `evaluate_rag.py` to perform a full evaluation with real answers and contexts.
* **Multi-hop Queries**: Extend the system to handle complex queries that require retrieving information from multiple sources or performing multiple retrieval steps.
* **Download JSON Results**: Add a feature to download the retrieved quotes and explanations as a JSON file.
* **Visualizations**: Incorporate visualizations of quote/author/tag distributions or similarity scores.
* **More LLM Options**: Allow users to select different LLMs (e.g., other Groq models, or integrate with other providers).
* **User Feedback**: Implement a mechanism for users to provide feedback on the relevance of retrieved quotes or explanations.

