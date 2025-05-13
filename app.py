from flask import Flask, render_template, request
import google.generativeai as genai
from embedding_utils import FAQRetriever

# Initialize Gemini Pro
genai.configure(api_key="AIzaSyAxX-hKYRS2Kjp0zFcdIzgavhZGjTX-jqM")
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

app = Flask(__name__)
retriever = FAQRetriever()

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_query = request.form["query"]
        context = retriever.retrieve(user_query)
        
        prompt = f"""You are a financial advisor in Uganda. 
Answer the user's question based on the FAQ knowledge below.

FAQ Knowledge:
{context}

User Question: {user_query}
Answer:"""

        result = model.generate_content(prompt)
        response = result.text

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
