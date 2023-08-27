# localGPT

This project is a fork of [localGPT](https://github.com/PromtEngineer/localGPT) (itself inspired by privateGPT). It use the latest models, and target computer without GPU, while still offering GPU support if you want.

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point.!

Built with [LangChain](https://github.com/hwchase17/langchain) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

Two configurations are provided, choose one:
* 13B: high quality answer, for higher-end computer, with at least 16GB of RAM and GPU recommanded.
  * Embeddings: [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) (MIT License)
  * LLM: [TheBloke/OpenOrca-Platypus2-13B-GGML](https://huggingface.co/TheBloke/OpenOrca-Platypus2-13B-GGML) (CC BY-NC-4.0 and Llama 2)
* 7B: good answer, for high-end computer, with 16GB of available RAM.
  * Embeddings: [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-large-en) (MIT License)
  * LLM: [TheBloke/orca_mini_v3_7B-GGML](https://huggingface.co/TheBloke/orca_mini_v3_7B-GGML) (Llama 2)

# Environment Setup

Install [mamba](https://github.com/mamba-org/mamba), a fast and open alternative to conda.
```shell
mamba create -p .env python=3.11 -c conda-forge
```

Activate
```shell
conda activate .env
```

In order to set your environment up to run the code here, first install all requirements:
```shell
pip install -r requirements.txt
```

Prompt evaluation is very slow on cpu, it's recommended to use a BLAS backend to speed things up.
If you want to use BLAS or Metal with [llama-cpp-python](<(https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal)>) you can set appropriate flags:
```shell
# Example: cuBLAS for NVIDIA GPU, required cuda-toolkit
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -r requirements.txt
```

## Docker

Installing the required packages for GPU inference on Nvidia GPUs, like gcc 11 and CUDA 11, may cause conflicts with other packages in your system.
As an alternative to Conda, you can use Docker with the provided Dockerfile.
It includes CUDA, your system just needs Docker, BuildKit, your Nvidia GPU driver and the Nvidia container toolkit.
Build as `docker build . -t localgpt`, requires BuildKit.
Docker BuildKit does not support GPU during *docker build* time right now, only during *docker run*.
Run as `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt`.

## Test dataset

This repo uses as an example [What are embeddings](https://vickiboykis.com/what_are_embeddings/) by Vicki Boykis, licensed under a Creative Commons, By Attribution, Non-Commercial, Share Alike 3.0 license.

## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, or .csv files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory.

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.

Run the following command to ingest all the data.

`defaults to cuda`

```shell
python ingest.py
```

Use the device type argument to specify a given device.

```sh
python ingest.py --device_type cpu
```

Use help for a full list of supported devices.

```sh
python ingest.py --help
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python run_localGPT.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Run it on CPU

By default, localGPT will use your GPU to run both the `ingest.py` and `run_localGPT.py` scripts. But if you do not have a GPU and want to run this on CPU, now you can do that (Warning: Its going to be slow!). You will need to use `--device_type cpu`flag with both scripts.

For Ingestion run the following:

```shell
python ingest.py --device_type cpu
```

In order to ask a question, run a command like:

```shell
python run_localGPT.py --device_type cpu
```

# Run quantized for M1/M2:

GGML quantized models for Apple Silicon (M1/M2) are supported through the llama-cpp library, [example](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML). GPTQ quantized models that leverage auto-gptq will not work, [see here](https://github.com/PanQiWei/AutoGPTQ/issues/133#issuecomment-1575002893). GGML models will work for CPU or MPS.

## Troubleshooting

**Install MPS:**
1- Follow this [page](https://developer.apple.com/metal/pytorch/) to build up PyTorch with Metal Performance Shaders (MPS) support. PyTorch uses the new MPS backend for GPU training acceleration. It is good practice to verify mps support using a simple Python script as mentioned in the provided link.

2- By following the page, here is an example of what you may initiate in your terminal

```shell
xcode-select --install
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install chardet
pip install cchardet
pip uninstall charset_normalizer
pip install charset_normalizer
pip install pdfminer.six
pip install xformers
```

**Upgrade packages:**
Your langchain or llama-cpp version could be outdated. Upgrade your packages by running install again.

```shell
pip install -r requirements.txt
```

If you are still getting errors, try installing the latest llama-cpp-python with these flags, and [see thread](https://github.com/abetlen/llama-cpp-python/issues/317#issuecomment-1587962205).

```shell
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```

# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   - Universal Windows Platform development
   - C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:

Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers.

# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. The LLM used are based on the Llama 2 model so they retain the original [Llama 2 license](https://github.com/facebookresearch/llama/blob/main/LICENSE).

# Common Errors

 - [Torch not compatible with CUDA enabled](https://github.com/pytorch/pytorch/issues/30664)

   -  Get CUDA version
      ```shell
      nvcc --version
      ```
      ```shell
      nvidia-smi
      ```
   - Try installing PyTorch depending on your CUDA version
      ```shell
         conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
      ```
   - If it doesn't work, try reinstalling
      ```shell
         pip uninstall torch
         pip cache purge
         pip install torch -f https://download.pytorch.org/whl/torch_stable.html
      ```

- [ERROR: pip's dependency resolver does not currently take into account all the packages that are installed](https://stackoverflow.com/questions/72672196/error-pips-dependency-resolver-does-not-currently-take-into-account-all-the-pa/76604141#76604141)
  ```shell
     pip install h5py
     pip install typing-extensions
     pip install wheel
  ```
- [Failed to import transformers](https://github.com/huggingface/transformers/issues/11262)
  - Try re-install
    ```shell
       conda uninstall tokenizers, transformers
       pip install transformers
    ```
