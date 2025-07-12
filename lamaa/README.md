to run 
python -m venv venv
venv\Scripts\activate

pip install fastapi uvicorn pydantic python-multipart

then run this 
uvicorn app:app --reload --port 8000

make sure you have static folder and app.py only









# Mental Health Chat Bot

A compassionate and intelligent mental health chatbot designed to assist users in managing their mental well-being. Built using Python and natural language processing capabilities, this chatbot offers meaningful conversations and resources to promote emotional health.

## Features
- **Compassionate Conversations**: The chatbot provides empathetic and non-judgmental responses to help users express their emotions and feel heard.
- **Resource Recommendations**: Offers guidance and resources, such as meditation practices, coping strategies, and mental health support organizations.
- **Anonymity and Privacy**: Ensures that all interactions are private and confidential, fostering a safe space for users.
- **Interactive and Adaptive**: Learns from interactions to provide more personalized and meaningful responses over time.

---

## Abstract
The Mental Health Chat Bot is an AI-driven virtual assistant designed to facilitate conversations that promote mental well-being. Leveraging Python and advanced NLP libraries like LangChain, this chatbot engages users in a supportive dialogue, offering a non-judgmental listening ear. In addition to conversational support, it provides helpful resources tailored to individual needs, such as relaxation techniques and contact information for professional help. The chatbot's primary aim is to offer a safe and confidential space for users to express their thoughts and feelings, making mental health care more accessible and less stigmatized.


![image](https://github.com/user-attachments/assets/ba8e8317-61f9-4af9-8459-bd3f565fbc25)

## Installation

### Step 1: Clone the Repository
To get started, clone this repository to your local machine:


### Step 2: Create a Python Virtual Environment
Creating a virtual environment is optional but recommended for managing dependencies:
bash
python -m venv venv
### Step 3: Activate the Virtual Environment
- On **Windows**:
  venv\Scripts\activate
### Step 4: Install Dependencies
Download and install all the required Python packages:
Copy code
pip install -r requirements.txt


### Step 5 Running Chainlit (if applicable)
If your project uses Chainlit, follow these steps:

### Step 6 Install Chainlit (if not already installed):
pip install chainlit
Run Chainlit:


chainlit run model.py -w

```bash
