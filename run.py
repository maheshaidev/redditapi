#!/usr/bin/env python3
"""
Redviddown API Server Startup Script
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Ensure directories exist
    Path("temp").mkdir(exist_ok=True)
    Path("downloads").mkdir(exist_ok=True)
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"Starting Redviddown API server on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        access_log=debug,
        log_level="info" if debug else "warning"
    )
