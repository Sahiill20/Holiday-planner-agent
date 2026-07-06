# ✈️ Personalized Holiday Management Agent

A premium, conversational travel planning application built using **LangGraph**, **LangChain**, and **Streamlit**. The agent utilizes real-time DuckDuckGo web searching to estimate pricing ranges (accommodation, food, transport) and verify landmark schedule constraints (e.g. weekly closures) on-the-fly.

---

## 🚀 Key Features

* **Multi-Agent Orchestration**: Powered by LangGraph's state machine including safety screeners, planners, schedule/geographic constraints validators, weather adaptors, and budget checkers.
* **Hugging Face Serverless Client**: Dynamically calls free, high-performance conversational LLMs like **Qwen 2.5 (72B)** or **Llama 3.1** via Hugging Face Cloud.
* **Real-time Web Search Enrichment**: Dynamically queries transport fares, hotel price ranges, and landmark schedules.
* **Dynamic Weather Adaptation**: Queries the Open-Meteo API for real-time weather forecasts if travel dates fall within 14 days.
* **Model Selection Selector**: Swap LLM models on-the-fly from the Streamlit sidebar.

---

## 💻 Local Setup & Execution

1. Clone or copy this repository to your local directory.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root folder and add your Hugging Face API key:
   ```env
   HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended & Free)
Streamlit Community Cloud is the easiest way to deploy this app for free directly from your GitHub repository.

1. **Push your code to GitHub**:
   * Create a public repository on GitHub.
   * Initialize git locally and push this project:
     ```bash
     git init
     git add .
     git commit -m "Initial commit"
     git branch -M main
     git remote add origin <your-github-repo-url>
     git push -u origin main
     ```
2. **Deploy on Streamlit Cloud**:
   * Go to [share.streamlit.io](https://share.streamlit.io/) and log in (or sign up with your GitHub account).
   * Click **New app**.
   * Select your repository, branch (`main`), and set the main file path to `app.py`.
3. **Configure Environment Secrets**:
   * Before clicking deploy, click **Advanced settings**.
   * Under the **Secrets** section, paste your environment token:
     ```toml
     HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token_here"
     ```
   * Click **Save** and then click **Deploy**. Your app will build and give you a working public link!

---

### Option 2: Hugging Face Spaces (Free)
You can host it directly on Hugging Face Spaces for free:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your space, select **Streamlit** as the SDK, and choose the Free tier.
3. Once the Space is created, navigate to **Settings** -> **Variables and secrets** -> **New secret**.
4. Add a secret named `HUGGINGFACEHUB_API_TOKEN` with your active token.
5. Clone the space repository locally or upload your files (`app.py`, `requirements.txt`, and the `src/` folder) directly using the Hugging Face web interface.
6. Hugging Face will automatically build and serve the application!
