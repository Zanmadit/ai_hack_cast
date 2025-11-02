# AI Trends Forecasting Dashboard

A machine learning system that analyzes historical AI industry data and forecasts future trends using LSTM neural networks. The project provides interactive visualizations showing both historical patterns and 5-year predictions across 19 different AI-related metrics.

## What It Does

This system tracks and predicts the evolution of AI technology by analyzing:
- AI model characteristics (size, training compute, citations)
- Infrastructure metrics (GPU clusters, hardware capabilities)
- Cost trends (training costs, hardware expenses)

It uses time series forecasting to help understand where the AI industry is heading based on historical data from 2010 onwards.

## Key Features

- **LSTM-based Forecasting**: 19 pre-trained models that predict future trends with 5-year horizons
- **Interactive Visualizations**: Side-by-side comparison of historical data vs. forecasted trends
- **Multiple Metrics**: Tracks model parameters, training compute (FLOPs), citations, costs, GPU cluster rankings, and hardware performance
- **REST API**: FastAPI backend serving forecast data and visualizations
- **Modern Frontend**: React-based dashboard with Plotly charts

## Tech Stack

### Backend
- **FastAPI** - REST API framework
- **TensorFlow/Keras** - LSTM model training and inference
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Data preprocessing (MinMaxScaler)
- **Plotly** - Server-side graph generation
- **Joblib** - Model serialization

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool
- **Plotly.js** - Interactive visualizations
- **Axios** - HTTP client

## How It Works

```
┌─────────────────────────────────────────┐
│         React Frontend (Vite)           │
│  Interactive dashboard with Plotly      │
└──────────────┬──────────────────────────┘
               │ HTTP GET /graphs
┌──────────────▼──────────────────────────┐
│      FastAPI Backend (Python)           │
│  • Load pre-trained LSTM models         │
│  • Aggregate historical data by year    │
│  • Generate 5-year forecasts            │
│  • Create Plotly visualizations         │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌─────▼─────┐
│  CSV   │          │   LSTM    │
│  Data  │          │  Models   │
│  (6)   │          │  (.keras) │
└────────┘          └───────────┘
```

**Pipeline:**
1. Historical data is loaded from CSV files (2010+)
2. Data is aggregated by year using median values
3. Pre-trained LSTM models generate forecasts for next 5 years
4. Plotly creates interactive graphs showing historical + forecast data
5. Frontend fetches and displays all visualizations

## Project Structure

```
ai_hack_cast/
├── data/                    # Historical datasets (6 CSV files)
│   ├── all_ai_models.csv
│   ├── frontier_ai_models.csv
│   ├── gpu_clusters.csv
│   └── ml_hardware.csv
│
├── models/                  # Pre-trained LSTM models (19 models + scalers)
│   ├── *_model.keras
│   └── *_scaler.pkl
│
├── notebooks/               # Jupyter notebooks for training and experiments
│   ├── lstm_model.ipynb    # Model training pipeline
│   ├── doc_retrieval.ipynb # PDF search (experimental)
│   └── get_arxiv_data.ipynb
│
├── src/
│   └── main.py             # FastAPI backend server
│
└── frontend/               # React application
    ├── src/
    │   └── App.jsx        # Main dashboard component
    └── package.json
```

## Setup & Installation

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install fastapi uvicorn tensorflow pandas numpy scikit-learn plotly joblib
```

3. Start the FastAPI server:
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The dashboard will be available at `http://localhost:5173`

## Usage

1. Start both backend and frontend servers (see Setup section)
2. Open your browser to `http://localhost:5173`
3. The dashboard will automatically fetch forecast data from the API
4. Explore 19 different metrics with interactive Plotly charts
5. Each metric shows historical data (left) vs. 5-year forecast (right)

## Metrics Tracked

The system forecasts 19 different metrics across multiple datasets:

**AI Models:**
- Model parameters (size)
- Training compute (FLOPs)
- Citations count
- Training costs

**GPU Clusters:**
- Power capacity
- Rankings
- Total cluster metrics

**Hardware:**
- Performance benchmarks
- Power consumption (TDP)
- Hardware costs

Each metric is forecasted independently using its own trained LSTM model.

## Model Training

To retrain models, use the Jupyter notebook at `notebooks/lstm_model.ipynb`:
- Loads historical data from CSV files
- Creates 3-year sliding window sequences
- Applies log transformation + MinMax scaling
- Trains LSTM (32 units, 0.1 dropout, early stopping)
- Saves models to `models/` directory

## Document Retrieval (Experimental)

The `notebooks/doc_retrieval.ipynb` notebook implements a PDF document retrieval system using ColBERT v2.0 for semantic search:

### Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  PDF File   │────>│ Text Extract │────>│  Chunking   │────>│  ColBERT │
│             │     │   (PyMuPDF)  │     │  (overlap)  │     │  Indexer │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
                                                                     │
                                                                     v
                                                          ┌──────────────────┐
                                                          │ Indexed Vectors  │
                                                          │   (searchable)   │
                                                          └──────────────────┘
```

### Components

**1. PDF Text Extraction**
- Uses PyMuPDF (fitz) to extract text from PDF files
- Cleans extracted text by normalizing whitespace
- Handles multi-page documents

**2. Text Chunking**
- Splits documents into overlapping chunks (default: 512 words)
- Overlap of 128 words between chunks to preserve context
- Each chunk stored with metadata (chunk_id, source, preview)

**3. ColBERT v2.0 Indexing**
- Uses `colbert-ir/colbertv2.0` checkpoint
- Configuration:
  - Document max length: 512 tokens
  - Query max length: 128 tokens
  - Embedding dimension: 256
  - Similarity metric: cosine
  - Compression: 2-bit quantization
- GPU acceleration when available (CUDA)

**4. Semantic Search**
- Late interaction mechanism for efficient retrieval
- Returns top-k relevant passages with scores
- Results include passage ID, rank, and relevance score

### Usage Example

```python
# Extract and index a PDF
pdf_file = "document.pdf"
index_name, metadata = embed_pdf_with_colbert(pdf_file)

# Search the indexed document
query = "What are the key findings?"
results = search_pdf(query, index_name, k=5)

# Results contain ranked passages with relevance scores
for result in results:
    print(f"Rank {result['rank']}: Score {result['score']}")
```

### Dependencies
```bash
pip install pymupdf colbert-ai torch
```

## API Endpoints

**GET /graphs**
- Returns JSON with 19 forecast visualizations
- Each contains historical and forecast Plotly graphs
- Response format:
```json
{
  "graphs": [
    {
      "name": "metric_name",
      "historical_graph": {...},
      "forecast_graph": {...}
    }
  ]
}
```

## Requirements

- Python 3.8+
- Node.js 18+
- 8GB+ RAM (for loading models and processing data)
- Modern browser with JavaScript enabled
