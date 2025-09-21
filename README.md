# LVX-Deal-Spotter-RAG-Processor

This repository contains the code for the RAG (Retrieval-Augmented Generation) processor for the LVX Deal Spotter application. This service is a Flask application responsible for processing documents, extracting text, generating embeddings, and storing them in Google Cloud's Vertex AI Vector Search.

## Features

- Extracts text from various file formats including PDF, DOCX, PPTX, images, and audio files.
- Generates embeddings for text chunks using Google's Vertex AI.
- Stores embeddings and metadata in Vertex AI Vector Search for efficient similarity search.
- Integrates with Firebase Realtime Database to track the status of document processing.
- Provides a health check endpoint.

## Environment Variables

The following environment variables are required to run the application:

- `PROJECT_ID`: Your Google Cloud project ID.
- `GCP_REGION`: The Google Cloud region for your resources (e.g., `us-central1`).
- `EMBEDDING_MODEL`: The name of the Vertex AI embedding model to use (e.g., `textembedding-gecko@003`).
- `INDEX_ID`: The ID of your Vertex AI Vector Search index.
- `STAGING_BUCKET_URI`: The Google Cloud Storage URI for staging data (e.g., `gs://your-staging-bucket`).

## API Endpoints

### `/health`

- **Method:** `GET`
- **Description:** Returns the health status of the application.
- **Response:** `{"status": "healthy"}`

### `/process`

- **Method:** `POST`
- **Description:** Triggers the processing of a document.
- **Request Body:**
  ```json
  {
    "doc_id": "unique-document-id",
    "bucket": "your-gcs-bucket",
    "filename": "path/to/your/file.pdf",
    "deal_id": "associated-deal-id"
  }
  ```
- **Success Response:** `{"status": "success", "doc_id": "unique-document-id"}`
- **Error Response:** `{"status": "failed", "doc_id": "unique-document-id"}`, status code 500.

## Dependencies

Key Python libraries used:

- Flask
- Firebase Admin SDK
- Google Cloud Storage
- Google Cloud AI Platform
- LangChain
- PyPDF2, python-docx, python-pptx for file parsing
- Pytesseract for OCR
- Gunicorn for production server

For a full list of dependencies, see `requirements.txt`.

## Running Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set environment variables:**
    ```bash
    export PROJECT_ID="your-project-id"
    export GCP_REGION="your-gcp-region"
    export EMBEDDING_MODEL="your-embedding-model"
    export INDEX_ID="your-index-id"
    export STAGING_BUCKET_URI="gs://your-staging-bucket"
    ```
3.  **Run the application:**
    ```bash
    flask run --port 8080
    ```

## Docker

To build and run the application as a Docker container:

1.  **Build the image:**
    ```bash
    docker build -t lvx-deal-spotter-rag-processor .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8080:8080 \
      -e PROJECT_ID="your-project-id" \
      -e GCP_REGION="your-gcp-region" \
      -e EMBEDDING_MODEL="your-embedding-model" \
      -e INDEX_ID="your-index-id" \
      -e STAGING_BUCKET_URI="gs://your-staging-bucket" \
      lvx-deal-spotter-rag-processor
    ```
