# BI Agent Core

This directory contains the core logic for the Business Intelligence AI Agent.

## Running the Chainlit App

To run the Chainlit app:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the `.env.example` file to `.env` and fill in your database credentials:
   ```bash
   cp .env.example .env
   ```

3. (Optional) Initialize the database with sample data:
   ```bash
   python init_db.py
   ```

4. Run the Chainlit app:
   ```bash
   chainlit run app.py
   ```

The app will start on http://localhost:8000 by default.

## Development Setup

To install the project in editable mode, run the following command from the project root:

```bash
pip install -e .
```