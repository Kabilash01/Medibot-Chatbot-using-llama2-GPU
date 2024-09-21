

# Medibot Chatbot using LLaMA2-GPU



## About the Project

*Medibot* is an AI-powered chatbot designed to provide medical-related information and support. The chatbot is built using the *LLaMA2* model, optimized for running on *GPU*, which makes it capable of handling complex medical queries efficiently and in real time. The system leverages large language models (LLMs) to provide accurate and contextually appropriate responses for healthcare applications.

## Features
- Natural language understanding to interpret medical queries.
- Runs on GPU for high performance and fast inference.
- Built on the LLaMA2 model for state-of-the-art language processing.
- Easily extensible for integrating with medical knowledge databases or APIs.
- Scalable deployment options for local and cloud environments.

## Getting Started

### Prerequisites
To set up and run the project locally, you will need:
- Python 3.8 or higher
- CUDA-enabled GPU for optimal performance
- [PyTorch](https://pytorch.org/) installed with GPU support
- [LLaMA2](https://huggingface.co/meta-llama/LLaMA2) model weights
- Basic knowledge of machine learning and Python

### Installation

1.Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

   

### Download the LLaMA2 model weights and place them in the appropriate directory. You may need to sign up for access on Hugging Face.
   https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML

###Ensure your CUDA drivers are installed and configured for PyTorch to utilize GPU.
   #recommended to use CUDA 12.4

### Usage

1. Run the chatbot:
   
```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
   

2. The chatbot interface will start, and you can begin interacting with Medibot by asking medical-related questions.

3. To deploy the chatbot on the web or in a production environment, follow the deployment steps in the deployment.md file (if available).

## How It Works

Medibot uses the *LLaMA2* large language model, which is fine-tuned for medical conversations. The chatbot operates by:
1. *Input Processing*: Natural language queries are parsed and converted into a format suitable for the LLaMA2 model.
2. *Model Inference*: The LLaMA2 model, running on a GPU, processes the query and generates a context-aware response.
3. *Response Generation*: The chatbot returns a human-like response with relevant medical information.

For more technical details, refer to the [architecture documentation](docs/architecture.md).

## Technologies Used
- *Python*: Core programming language.
- *LLaMA2*: Large language model from Meta.
- *PyTorch*: Deep learning framework for model training and inference.
- *CUDA*: GPU acceleration for high-performance computing.
- *Hugging Face*: Platform for sharing and deploying AI models.
  
## Contributing

We welcome contributions from the open-source community! To contribute:
1. Fork the project.
2. Create a new branch for your feature (git checkout -b feature/new-feature).
3. Commit your changes (git commit -m 'Add new feature').
4. Push to your branch (git push origin feature/new-feature).
5. Open a Pull Request.

## License

Distributed under the MIT License. See LICENSE for more information.

---

This draft README provides a basic overview of the project, guides for setup and usage, and includes links to additional documentation where relevant. Make sure to update any links to images, documentation, and additional files that are specific to the repository.
