import logging

import click
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_NAME, MODEL_MAX_CTX_SIZE, MODEL_STOP_SEQUENCE, MODEL_PROMPT_TEMPLATE, MODEL_GPU_LAYERS


def load_model():
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        conf (Config): a localGPT config dataclass

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {MODEL_ID}, with {MODEL_GPU_LAYERS} gpu layers")
    logging.info("This action can take a few minutes!")

    if MODEL_NAME is not None:
        if ".ggml" in MODEL_NAME:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_NAME)
            max_ctx_size = MODEL_MAX_CTX_SIZE
            max_token = min(512, MODEL_MAX_CTX_SIZE) # we limit the amount of generated token. We want the answer to be short.
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_token, 
            }
            kwargs["n_gpu_layers"] = MODEL_GPU_LAYERS
            kwargs["n_threads"] = 8
            kwargs["stop"] = MODEL_STOP_SEQUENCE
            kwargs["n_batch"] = 512 # faster prompt evaluation. It's important to speed it up because it contain the context
            kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]
            kwargs["temperature"] = 0.4 # default is 0.8, values between 0 and 1 doesn't affect much the result
            return LlamaCpp(**kwargs)

    raise NotImplemented("This version of localGPT only support GGML model")


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=MODEL_PROMPT_TEMPLATE)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    llm = load_model()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        print()
        docs = res["source_documents"]

        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"])
                print(document.page_content)
        else:
            sources = set([d.metadata["source"] for d in docs])
            for s in sources:
                print(f"> {s}")
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.ERROR
    )
    main()
