import os
import fitz  
from dotenv import load_dotenv  # 🔐 Add this
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ Load .env
load_dotenv()

# ✅ Fetch from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set Google API key for embeddings
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Groq AI LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Student-friendly prompt
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful academic assistant. Use the provided document context along with your own knowledge to answer the question clearly and simply.

    - If the question relates to a topic in the context, prioritize it.
    - If something isn't in the context, explain using external knowledge.
    - Use examples or step-by-step explanations when possible.
    - Be concise and student-friendly.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# FAISS vector store
vectors = None
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def process_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def store_embeddings(chunks):
    global vectors
    vectors = FAISS.from_texts(chunks, embeddings)

def retrieve_answer(question):
    if vectors is None:
        return "No document data available. Please upload a PDF first."

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': question})
    return response['answer']

def mains(request):
    global vectors

    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        question = request.POST.get('question')

        if uploaded_file:
            file_path = f"media/uploads/{uploaded_file.name}"
            default_storage.save(file_path, ContentFile(uploaded_file.read()))

            chunks = process_pdf(file_path)
            store_embeddings(chunks)

        answer = retrieve_answer(question) if question else ""
        return render(request, "P1.HTML", {'question': question, 'answer': answer})

    return render(request, "P1.HTML")
