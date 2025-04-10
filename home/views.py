import os
import io
import fitz  # PyMuPDF
from dotenv import load_dotenv
from django.shortcuts import render
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set Google API key for embeddings
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Global FAISS vector store
vectors = None

# Student-friendly prompt
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful academic assistant. Use the provided document context along with your own knowledge to answer the question clearly and simply.

- If the question relates to a topic in the context, prioritize it.
- If something isn't in the context, explain using external knowledge.
- Use examples or step-by-step explanations when possible.
- Be concise and student-friendly.
                                                   
Generate output using ONLY valid HTML tags:
- Use <h3> for section titles
- Use <ol><li><p>... </p></li></ol> for lists
- Do NOT use any asterisks (*), dashes (-), or Markdown formatting
- Insert <br> after every <li> and <p>
- Do not add empty <li> or <p> tags
- Avoid extra whitespace or empty bullets

<context>
{context}
</context>

Question: {input}
""")

# Extract & split PDF text
def process_pdf(file_obj):
    doc = fitz.open(stream=file_obj, filetype="pdf")  # Open from memory
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Store vector embeddings
def store_embeddings(chunks):
    global vectors
    vectors = FAISS.from_texts(chunks, embeddings)

# Retrieve answer from vectorstore
def retrieve_answer(question):
    if vectors is None:
        return "No document uploaded. Please upload a PDF first."

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    result = retrieval_chain.invoke({"input": question})
    return result["answer"]

# Main view with Reset support
def mains(request):
    global vectors
    file_name = None
    is_file_loaded = False

    if request.method == "POST":
        #  Handle Reset
        if request.POST.get("reset") == "true":
            vectors = None
            request.session.flush()
            return render(request, "P1.HTML", {
                "question": "",
                "answer": "",
                "file_name": None,
                "is_file_loaded": False
            })

        uploaded_file = request.FILES.get("file")
        question = request.POST.get("question", "")
        answer = ""

        if uploaded_file:
            pdf_bytes = uploaded_file.read()
            chunks = process_pdf(io.BytesIO(pdf_bytes))
            store_embeddings(chunks)
            file_name = uploaded_file.name  # Get uploaded name
            is_file_loaded = True

            # Store filename in session
            request.session["file_name"] = file_name
            request.session["is_file_loaded"] = True

        elif vectors is not None:
            # 👇 Retrieve from session if file not re-uploaded
            file_name = request.session.get("file_name", "Uploaded PDF")
            is_file_loaded = request.session.get("is_file_loaded", False)

        if question and vectors is not None:
            answer = retrieve_answer(question)

        return render(request, "P1.HTML", {
            "question": question,
            "answer": answer,
            "file_name": file_name,
            "is_file_loaded": is_file_loaded
        })

    # GET request — show remembered state if available
    file_name = request.session.get("file_name")
    is_file_loaded = request.session.get("is_file_loaded", False)

    return render(request, "P1.HTML", {
        "question": "",
        "answer": "",
        "file_name": file_name,
        "is_file_loaded": is_file_loaded
    })
