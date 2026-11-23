from fastapi import FastAPI

app = FastAPI(title="MP231 ATM Demand API")


@app.get("/health")
def health():
    return {"status": "ok"}
