from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Financial_News_Analysis_Pipeline import run_pipeline
import json
import os
from datetime import datetime, timedelta
import asyncio
from threading import Thread
import time
import uvicorn

app = FastAPI()

# Add CORS middleware to fix React fetch errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for storing results and metadata
cache = {
    "sentiment_results": None,
    "daily_sentiment_index": None,
    "last_updated": None,
    "is_running": False
}

# Configuration
CACHE_DURATION_HOURS = 1  # Run pipeline max once per hour
RESULTS_DIR = "Financial_News_Data"

def load_results_from_file(filename):
    """Load results from JSON file if it exists"""
    filepath = os.path.join(RESULTS_DIR, filename)
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
    return None

def should_run_pipeline():
    """Check if pipeline should run based on cache expiry"""
    if cache["last_updated"] is None:
        return True
    
    time_since_update = datetime.now() - cache["last_updated"]
    return time_since_update > timedelta(hours=CACHE_DURATION_HOURS)

def run_pipeline_background():
    """Run pipeline in background thread"""
    if cache["is_running"]:
        print("Pipeline already running, skipping...")
        return
    
    cache["is_running"] = True
    try:
        print("Running pipeline...")
        run_pipeline()
        
        # Load results into cache
        cache["sentiment_results"] = load_results_from_file("Sentiment_Analysis_Results.json")
        cache["daily_sentiment_index"] = load_results_from_file("Daily_Sentiment_Index.json")
        cache["last_updated"] = datetime.now()
        
        print("Pipeline completed successfully")
    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        cache["is_running"] = False

# Initialize cache on startup
@app.on_event("startup")
async def startup_event():
    # Try to load existing results first
    cache["sentiment_results"] = load_results_from_file("Sentiment_Analysis_Results.json")
    cache["daily_sentiment_index"] = load_results_from_file("Daily_Sentiment_Index.json")
    
    # If no cached results exist, run pipeline once
    if cache["sentiment_results"] is None or cache["daily_sentiment_index"] is None:
        # Run in background thread to not block startup
        thread = Thread(target=run_pipeline_background)
        thread.daemon = True
        thread.start()

@app.get("/sentiment")
async def get_sentiment_analysis():
    # Check if we need to refresh data
    if should_run_pipeline() and not cache["is_running"]:
        # Start background update
        thread = Thread(target=run_pipeline_background)
        thread.daemon = True
        thread.start()
    
    # Return cached results if available
    if cache["sentiment_results"] is not None:
        return JSONResponse(content={
            "data": cache["sentiment_results"],
            "last_updated": cache["last_updated"].isoformat() if cache["last_updated"] else None,
            "is_updating": cache["is_running"]
        })
    
    # If no cached results and not running, try loading from file
    results = load_results_from_file("Sentiment_Analysis_Results.json")
    if results:
        return JSONResponse(content={
            "data": results,
            "last_updated": None,
            "is_updating": cache["is_running"]
        })
    
    # If still no results
    if cache["is_running"]:
        raise HTTPException(status_code=202, detail="Analysis is running, please try again in a few moments")
    else:
        raise HTTPException(status_code=404, detail="No sentiment analysis results available")

@app.get("/daily-sentiment-index")
async def get_daily_sentiment_index():
    # Check if we need to refresh data
    if should_run_pipeline() and not cache["is_running"]:
        # Start background update
        thread = Thread(target=run_pipeline_background)
        thread.daemon = True
        thread.start()
    
    # Return cached results if available
    if cache["daily_sentiment_index"] is not None:
        return JSONResponse(content={
            "data": cache["daily_sentiment_index"],
            "last_updated": cache["last_updated"].isoformat() if cache["last_updated"] else None,
            "is_updating": cache["is_running"]
        })
    
    # If no cached results and not running, try loading from file
    results = load_results_from_file("Daily_Sentiment_Index.json")
    if results:
        return JSONResponse(content={
            "data": results,
            "last_updated": None,
            "is_updating": cache["is_running"]
        })
    
    # If still no results
    if cache["is_running"]:
        raise HTTPException(status_code=202, detail="Analysis is running, please try again in a few moments")
    else:
        raise HTTPException(status_code=404, detail="No daily sentiment index results available")

@app.get("/status")
async def get_status():
    """Get current status of the analysis pipeline"""
    return JSONResponse(content={
        "is_running": cache["is_running"],
        "last_updated": cache["last_updated"].isoformat() if cache["last_updated"] else None,
        "has_sentiment_data": cache["sentiment_results"] is not None,
        "has_index_data": cache["daily_sentiment_index"] is not None
    })

@app.post("/refresh")
async def force_refresh():
    """Force refresh of analysis data"""
    if cache["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    
    # Start background update
    thread = Thread(target=run_pipeline_background)
    thread.daemon = True
    thread.start()
    
    return JSONResponse(content={"message": "Refresh started"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)