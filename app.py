import sys
import io
import uvicorn
from server.app import app

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("Launching the Autonomous CEO AI Simulator (Hackathon Edition)!")
    print("Running OpenEnv FastApi backend + Gradio Dashboard on port 7860...")
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
