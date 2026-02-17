# LangExtract Standalone Demo

This project demonstrates how to use `LangExtract` to extract structured data from unstructured text using Google's Gemini models.

## Prerequisites

1.  **Python 3.10+**
2.  A **Google Cloud API Key** (for Gemini).

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your environment variables:
    - Create a `.env` file in this directory.
    - Add your API key:
      ```
      GOOGLE_API_KEY=your_api_key_here
      ```

## Running the Demo

1.  View the sample data:
    - Open `sample_data.txt` to see the unstructured text.

2.  Run the extractor:
    ```bash
    python extract.py
    ```

3.  Check the output:
    - The script will print the structured JSON extraction to the console.
