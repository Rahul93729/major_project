from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model import qa_bot
import asyncio
import os
import subprocess

app = FastAPI()

# Enable CORS with correct configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",  # Add this for local development
        "http://localhost:3000",   # If you're using React default port
        "http://127.0.0.1:3000" 
           # Add more origins as needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specify allowed HTTP methods
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount the static directory
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

# Global variable for the chatbot
global_qa_chain = None

@app.on_event("startup")
async def startup_event():
    global global_qa_chain
    try:
        global_qa_chain = qa_bot()
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        # We'll initialize it on the first request if it fails here

@app.get("/")
async def read_root():
    response = {"status": "API is running"}
    return response

@app.get("/emotion")
async def emotion():
    try:
        # Adjust this command based on how you run the emotion detection script
        subprocess.Popen(["python", "../ee/kerasmodel.py", "--mode", "display"])
        return {"message": "Launching Emotion Detection..."}
    except Exception as e:
        return {"error": f"Failed to start emotion detection: {str(e)}"}

@app.post("/chat")
async def chat(request: ChatRequest):
    global global_qa_chain

    # Initialize qa_chain if not already done
    if global_qa_chain is None:
        try:
            global_qa_chain = qa_bot()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize chatbot: {str(e)}"
            )

    try:
        # Add some basic input validation
        if not request.message.strip():
            return {"response": "Please enter a valid message."}

        # Check if the message is related to mental health
        if "suicide" in request.message.lower():
            # Provide mental health resources
            return {
                "response": "I'm sorry you're struggling with your mental health. Here are some helpful resources:\n\n"
                "- National Suicide Prevention Lifeline: 1-800-273-8255 (available 24 hours everyday)\n"
                "- Crisis Text Line: Text HOME to 741741 to connect with a crisis counselor\n"
                "- Find a therapist: https://www.psychologytoday.com/us/therapists\n"
                "- Self-care tips: https://www.mind.org.uk/information-support/tips-for-everyday-living/\n"
                "Please don't hesitate to reach out for support. Your mental health is important."
            }

        response = await global_qa_chain.ainvoke(
            {"query": request.message}
        )

        result = response.get("result", "I couldn't generate an answer. Please try again.")
        return {"response": result}

    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return {
            "response": "I apologize, but I'm having trouble processing your request. Please try again in a moment."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)