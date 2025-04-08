import os
import fitz  
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


GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDomzgvWGihsCKBR1gGsr3xZlcs-Qw0IGE"

# Set Google API key for embeddings
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Groq AI LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Define Prompt Template for AI Response
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question using the provided context only.
    Ensure the response is accurate and relevant.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Store FAISS vectors globally
vectors = None
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def process_pdf(file_path):
    """Extract text from the PDF and split it into chunks."""
    doc = fitz.open(file_path)  # Open the uploaded PDF
    text = ""
    
    # Extract text from all pages
    for page in doc:
        text += page.get_text("text") + "\n"
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    return chunks

def store_embeddings(chunks):
    """Convert text chunks to embeddings and store them in FAISS."""
    global vectors  # Use a global variable to retain FAISS index

    # Create FAISS vector store
    vectors = FAISS.from_texts(chunks, embeddings)

def retrieve_answer(question):
    """Retrieve the most relevant document chunks and generate an answer using Groq AI."""
    if vectors is None:
        return "No document data available. Please upload a PDF first."

    # Create retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get AI response
    response = retrieval_chain.invoke({'input': question})

    return response['answer']

def mains(request):
    global vectors  # Ensure FAISS vectors persist

    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.FILES.get('file')
        question = request.POST.get('question')

        if uploaded_file:
            # Save uploaded file
            file_path = f"media/uploads/{uploaded_file.name}"
            default_storage.save(file_path, ContentFile(uploaded_file.read()))

            # Process PDF & store embeddings
            chunks = process_pdf(file_path)
            store_embeddings(chunks)

        # Process user question
        answer = retrieve_answer(question) if question else ""

        return render(request, "P1.HTML", {'question': question, 'answer': answer})

    return render(request, "P1.HTML")
