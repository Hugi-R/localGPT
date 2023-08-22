import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredHTMLLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses
# You also have to set various parameter for this model
MODEL_ID = "TheBloke/OpenOrca-Platypus2-13B-GGML"
MODEL_NAME = "openorca-platypus2-13b.ggmlv3.q4_K_M.bin"
MODEL_MAX_CTX_SIZE = 4096
MODEL_STOP_SEQUENCE = ["###"]
MODEL_GPU_LAYERS = 0
MODEL_PROMPT_TEMPLATE = """### Instruction:
Use the following pieces of Context to answer the Question at the end. Take into acount the conversation History.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

History:
{history}

Question:
{question}

### Response:
"""
