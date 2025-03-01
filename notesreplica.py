
import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import base64
import tempfile
from pdf2image import convert_from_path
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import torch

dotenv.load_dotenv()
qwen_api_key = os.getenv("QWEN_API_KEY")
deep_seek_api = os.getenv("DEEP_SEEK_API")
deep_llama_api = os.getenv("DEEP_LLAMA_API")
groq_api_key = os.getenv("groq_api_key")

def encode_image_to_base64(image_path):
    """Read a local image and return a base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF by converting pages to images and using OCR."""

    # Create temporary directory to store images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        split_images = []
        for i, image in enumerate(images):
            width, height = image.size
            
            # Split into left and right halves
            left = image.crop((0, 0, width // 2, height))
            right = image.crop((width // 2, 0, width, height))
            
            # Save split images
            # left_path = f"/page_{i + 1}_left.png"
            # right_path = f"/page_{i + 1}_right.png"
            
            # left.save(left_path)
            # right.save(right_path)
            
            split_images.extend([left, right])
            
            # Initialize HuggingFace embeddings model
            embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Process each image and create Document objects
        documents = []
        
        for i, image in enumerate(split_images):
            # Save image temporarily
            temp_image_path = os.path.join(temp_dir, f'page_{i}.jpg')
            image.save(temp_image_path, 'JPEG')
            
            # Convert image to base64
            base64_image = encode_image_to_base64(temp_image_path)
            
            # Create LangChain Document with page info and image data
            documents.append(
                Document(
                    page_content=f"Page {i+1} of document: {pdf_path}",
                    metadata={
                        "page_num": i+1,
                        "source": pdf_path,
                        "base64_image": base64_image
                    }
                )
            )
        
        # Create FAISS vector store from the documents
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings_model)
        
        # Initialize vision model for OCR
        # chat = ChatOpenAI(
        #     model="qwen/qwen2.5-vl-72b-instruct:free",
        #     base_url="https://openrouter.ai/api/v1",
        #     openai_api_key=qwen_api_key,
        #     temperature=0.3
        # )
        chat = ChatGroq(groq_api_key = groq_api_key,
               model_name="llama-3.2-90b-vision-preview",
               max_tokens=1000)

        
        extracted_text = []
        
        # Perform similarity search to find relevant documents
        query = "What is this document about? give every detail"
        relevant_docs = vectorstore.similarity_search(query, k=6)
        # Process only the relevant documents to extract text from images
        messages = []
        for doc in relevant_docs:
            base64_image = doc.metadata["base64_image"]
            
            # Create message with image for the relevant document
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Please extract all text from this image and describe any visible content."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
            messages.append(message)
        
        # Process all messages through the LLM outside the loop
        if messages:
            for message in messages:
                response = chat.invoke([message])
                
                # Extract the text content from the response
                extracted_content = response.content
                
                # Add the extracted text to our results
                extracted_text.append(extracted_content)
        
        # Return the extracted text
        return extracted_text

# deepseek/deepseek-r1:free

# qwen-2.5-32b

# chat = ChatOpenAI(
#     model="deepseek/deepseek-r1-distill-llama-70b:free",
#     base_url="https://openrouter.ai/api/v1",
#     openai_api_key=deep_llama_api
# )
# chat = ChatOpenAI(
#     model="deepseek/deepseek-r1:free",
#     base_url="https://openrouter.ai/api/v1",
#     openai_api_key=deep_seek_api
# )
chat = ChatGroq(groq_api_key = groq_api_key,
               model_name="deepseek-r1-distill-llama-70b",
               max_tokens=10000)


test_image_path = "/Users/ayushbhakat/Documents/Sem-6/Notes/page_1_left.png"
# Use provided image path or a sample base64 image
if test_image_path:
    # Read and encode the image file
    with open(test_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
else:
    # Sample base64 encoded image for testing (this is a placeholder)
    # In a real implementation, you would include an actual base64 string of a test image
    base64_image = "YOUR_BASE64_ENCODED_IMAGE_HERE"

# Create message with the image
# Get the extracted text from the PDF
import streamlit as st
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


st.title("Academic Notes Generator")
st.write("Upload a PDF to generate comprehensive academic study notes")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File uploaded successfully: {uploaded_file.name}")
    
    if st.button("Generate Notes"):
        with st.spinner("Extracting text from PDF..."):
            # Extract text from the uploaded PDF
            extracted_text = extract_text_from_pdf(temp_path)
            
            if not extracted_text:
                st.error("Could not extract text from the PDF. Please try another file.")
            st.info("Text extracted successfully. Generating notes...")
            
            # Define the prompt template for academic note creation
            prompt_template = ChatPromptTemplate.from_messages([
                ("human", """
                Here is the context extracted from the PDF: {extracted_text}
                
                You are an expert academic note-taking assistant. Your task is to analyze all text and image content extracted from the provided PDF and create comprehensive study notes.
                
                Please follow these guidelines:
                
                1. Organize the content into clear sections with proper headings and subheadings
                2. For each main topic:
                   - Provide a thorough explanation of concepts
                   - Include relevant examples from the source material
                   - Create visual representations where appropriate (tables, diagrams, flowcharts)
                   - Highlight key terms and definitions
                
                3. Ensure your explanations are detailed and comprehensive - leave nothing out from the source material
                
                4. After covering each section in depth, create:
                   - A concise point-wise summary of the main concepts
                   - Flashcard-style review notes with key questions and answers
                
                5. Conclude with a master summary that connects all the topics together
                
                Your notes should be both detailed enough for deep understanding and structured enough for efficient revision before examinations.
                """)
            ])
            
            # Create a chain that formats the prompt and passes it to the LLM
            chain = (
                {"extracted_text": RunnablePassthrough()} 
                | prompt_template 
                | chat
            )
            
            # Get the response
            message = chain.invoke(extracted_text)
            
            # Process the message through the LLM
            response = chat.invoke([message])
            
            # Display the generated notes
            st.subheader("Generated Study Notes")
            st.markdown(response.content)
            
            # Option to download the notes
            st.download_button(
                label="Download Notes",
                data=response.content,
                file_name=f"study_notes_{uploaded_file.name.split('.')[0]}.md",
                mime="text/markdown"
            )
    
    # Clean up the temporary file
    if os.path.exists(temp_path):
        if st.button("Clear"):
            os.remove(temp_path)
            st.success("Temporary files cleared")
