from fastapi import FastAPI
import time

app = FastAPI(title="FinDocAI", version="1.0.0", description="Intelligent Financial Document Processing API")

@app.get("/")
async def root():
    return {"message": "Welcome to FinDocAI - Intelligent Financial Document Processing API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)