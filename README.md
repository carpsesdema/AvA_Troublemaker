 AvA: Advanced Versatile Assistant
Your intelligent AI desktop partner, built on a flexible platform to supercharge complex tasks and workflows.

---

<p align="center">
  <img src="assets/AvA_Responding.gif" alt="AvA Demo - AI Assistant Responding" width="700">
</p>

---

## What is AvA?

In a world rapidly adopting Large Language Models (LLMs), effectively harnessing their power for complex, multi-step tasks directly on your desktop can be challenging. Managing different AI models, providing them with the right context from your local files, and orchestrating them for sophisticated workflows often requires significant boilerplate and specialized knowledge.

**AvA (Advanced Versatile Assistant)** is an intelligent desktop application designed by a solo developer to bridge this gap. It provides a rich, interactive environment where you can:

* **Integrate and chat with local LLMs** (via Ollama, supporting models like CodeLlama and Llama3) for privacy and offline capability.
* **Connect to powerful cloud-based models** like Google's Gemini API.
* **Leverage an advanced Retrieval Augmented Generation (RAG) system** that understands your local codebases, PDFs, and DOCX files, providing deep contextual awareness to the AI.
* **Execute innovative multi-step AI workflows**, such as the built-in multi-file code modification system where a "Planner AI" guides a "Specialist Generator AI."

At its core, AvA is more than just a collection of features; it's built as an **extensible AI platform**. The initial release focuses on supercharging developer workflows, but its underlying architecture—which combines a general-purpose conversational "Planner" AI with slots for specialized "Task" AIs and a robust RAG system—is designed for versatility. This opens the door for AvA to be adapted for a wide range of sophisticated AI-powered assistance across various domains in the future.

## The "Aha!" Moment: How AvA's Core Architecture Came To Be

As a developer, I often found myself needing to generate substantial amounts of code or understand large, existing codebases. While powerful cloud-based LLMs like Gemini are excellent for many tasks, direct API calls for massive code generation can become expensive very quickly.

My initial workflow involved:

1.  **Building a Knowledge Base:** I'd upload extensive Python code (and other documents) into a local RAG (Retrieval Augmented Generation) system. This gave a local language model (initially, a coding-specific Llama model via Ollama) a solid foundation to draw upon.
2.  **Interacting with the Code Model:** I then built an interface to interact with this local code model, leveraging the RAG for context.

However, a challenge emerged: effectively communicating complex needs or nuanced instructions directly to a specialized code model like CodeLlama isn't always straightforward. These models are fantastic at generation but might not excel at broader understanding or conversational dialogue.

**This led to the core innovation in AvA: introducing a general-purpose LLM (like Gemini or a capable Llama3 chat model) to act as an intelligent "translator" and "planner."**

This "Planner AI":
* Understands my natural language requests.
* Can break down complex tasks.
* Interacts with the RAG system for deep context.
* Then, intelligently prompts the specialized "Generator AI" (the local CodeLlama) to produce the precise code needed.

This hybrid approach allows AvA to combine the conversational strengths and planning capabilities of general LLMs with the cost-effective, specialized (and often private) code generation power of local models. It's about using the right AI for the right part of the job, all orchestrated seamlessly through a user-friendly desktop interface. This architecture also naturally paved the way for AvA to be more than just a coding tool, but a versatile platform for various AI-driven tasks.

## Key Features

AvA brings a suite of powerful AI capabilities to your desktop, designed and built to enhance complex workflows:

* **Hybrid LLM Integration:**
    * **Local LLMs via Ollama:** Run models like CodeLlama, Llama3, and others directly on your machine for privacy, offline access, and cost-effective generation.
    * **Gemini API Support:** Seamlessly switch to or complement local models with Google's powerful Gemini family for advanced tasks.
* **Advanced RAG (Retrieval Augmented Generation):**
    * Chat with your local codebase (Python files and more).
    * Index and query PDF and DOCX documents.
    * Utilizes FAISS for efficient vector storage/search and Langchain for orchestration.
* **Innovative Multi-File Code Modification:**
    * Go beyond single-file edits with an AI-driven workflow.
    * A "Planner AI" (e.g., Gemini) strategizes changes across multiple files.
    * A "Generator AI" (e.g., local CodeLlama) implements the code modifications based on the plan.
* **Extensible Platform Architecture:** AvA is built with modularity in mind. Its core design (Planner AI + Specialist AI + RAG) allows users to customize the AI setup for their specific needs and can be extended to support various AI models and tasks beyond the initial coding focus. The current model integrations (Ollama, Gemini) serve as a starting point for testing and demonstration. Future plans include broadening support for many more model backends (like OpenAI's GPT models and others) to give users maximum flexibility and choice.
* **User-Focused Desktop Experience:**
    * Developed with Python and PyQt6 for a responsive and native desktop feel.
    * Project-based organization to manage different contexts and knowledge bases.

## Current Status

AvA is currently in Alpha and is the result of a solo development effort. Your feedback and bug reports are highly valued and will directly contribute to its improvement!

## Getting Started (Alpha)

AvA is currently in Alpha. These instructions guide you on how to run it from the source code.

### Prerequisites

* **Python:** Version 3.11+ is recommended (AvA was developed using Python 3.13).
* **Git:** To clone the repository.
* **Ollama (Recommended for Local LLMs):**
    * If you wish to use local models like CodeLlama or Llama3, ensure Ollama is installed and running. Download it from [ollama.com](https://ollama.com/).
    * **Important:** After installing Ollama, you must pull the specific models you intend to use via the Ollama command line. For example:
        ```bash
        ollama pull codellama
        ollama pull llama3
        ```
        AvA will indicate if a selected Ollama model is not found locally.

### Installation & Running

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/carpsesdema/AvA_Troublemaker.git](https://github.com/carpsesdema/AvA_Troublemaker.git)
    cd AvA_Troublemaker
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Key for Gemini (Optional):**
    * To use Google Gemini models, you need an API key.
    * The application will look for an environment variable named `GEMINI_API_KEY`. You can set this in your terminal session:
        * Linux/macOS: `export GEMINI_API_KEY="YOUR_KEY_HERE"`
        * Windows (PowerShell): `$env:GEMINI_API_KEY="YOUR_KEY_HERE"`
    * Alternatively, you can create a `.env` file in the project's root directory (the same folder as `main.py`) and add the line: `GEMINI_API_KEY="YOUR_KEY_HERE"`.
    * If no key is provided, Gemini-based features will be unavailable, but local Ollama models will still function if Ollama is properly set up.

5.  **Run AvA:**
    * Ensure your virtual environment is activated.
    * Execute the main script from the project's root directory:
        ```bash
        python main.py
        ```

This should launch the AvA desktop application.

## Basic Usage / Quick Start

AvA is designed to be flexible. Here's a suggested workflow to get started and leverage its core capabilities:

1.  **Configure AI Persona (Optional but Recommended):**
    * Click on "Configure AI Persona" in the left panel.
    * Define a system prompt to guide the behavior of the primary chat LLM (e.g., Gemini or your chosen Ollama chat model). This helps tailor AvA's responses and conversational style.

2.  **Create a New Project:**
    * Click "Create New Project" in the left panel.
    * Give your project a name (e.g., `my_python_api`, `research_topic_xyz`). This creates an isolated context for your chats and RAG knowledge.

3.  **Start a New Chat:**
    * With your new project selected, click "New Chat." This opens a fresh chat tab associated with your project.

4.  **Build Your Knowledge Base (RAG):**
    * **For Code Projects:**
        * Use the "Add File(s)" or "Add Folder" buttons under your project's "KNOWLEDGE FOR '[Project Name]'" section in the left panel.
        * A powerful technique is to upload the `site-packages` directory from a Python virtual environment (`venv`) that contains all the libraries relevant to your current work. This gives the RAG system (and thus the AI) deep knowledge about the specific modules you're using.
    * **For Documents:**
        * Similarly, upload relevant PDFs or DOCX files to your project.
    * AvA will process and index these files, making their content available for contextual AI responses.

5.  **Interact with AvA:**
    * **Chat:** Ask questions, request explanations, or brainstorm ideas. If RAG is populated for the project, AvA will use that context.
    * **Code Generation/Modification:** For coding tasks, you can describe what you need. AvA's multi-file modification workflow can be triggered by describing changes to existing (RAG-indexed) code or requesting new files.
    * **Select Models:** Use the "Model" dropdown in the left panel to choose between configured Gemini models or available Ollama models for your chat interactions.

This workflow allows you to tailor AvA's knowledge and conversational style to your specific needs, making it a powerful and personalized AI partner.

## Technologies Used

AvA is built with a powerful stack of Python libraries and tools:

* **Core Application:**
    * Python 3.11+
    * PyQt6 for the native desktop graphical user interface.
    * qasync for integrating asyncio with PyQt6.
* **LLM Integration:**
    * `google-generativeai` for interacting with Google Gemini API models.
    * `ollama` client library for seamless communication with local Ollama instances (supporting models like CodeLlama, Llama3, etc.).
* **RAG System:**
    * `faiss-cpu` for efficient similarity search in vector stores.
    * `sentence-transformers` for generating text embeddings.
    * `langchain-text-splitters` (specifically `RecursiveCharacterTextSplitter` and `PythonCodeTextSplitter`) for intelligent document chunking.
    * `PyPDF2` for PDF document parsing.
    * `python-docx` for DOCX document parsing.
* **UI Enhancements & Utilities:**
    * `Markdown` for rendering rich text in chat.
    * `Pillow` for image handling.
    * `qtawesome` for easier icon integration in the UI.
    * `python-dotenv` for managing environment variables (like API keys).
* **Development & Structure:**
    * Organized into core services (session management, file uploads, vector database interactions, code analysis, image handling, document chunking), backend adapters for different LLMs, UI components, and utility modules.
    * Features extensive logging for diagnostics and debugging.

## How to Contribute

AvA is currently an alpha release developed by a solo developer. Feedback, bug reports, and feature suggestions are highly welcome and invaluable at this stage!

* **Reporting Issues:** If you encounter any bugs or unexpected behavior, please open an issue on the [GitHub Issues page](https://github.com/carpsesdema/AvA_Troublemaker/issues).
* **Feature Requests:** Have an idea that could make AvA even better? Feel free to submit it as a feature request on the Issues page.

As the project matures, guidelines for code contributions may be established.

## Support AvA's Development

AvA is a passion project. If you find it useful or believe in its vision, please consider supporting its continued development:

* [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/snowballKori)

Your support helps cover development costs and allows for more time to be dedicated to improving AvA.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contact & Links

* **Website / Landing Page:** [snowballannotation.com](http://snowballannotation.com)
* **GitHub Repository:** [https://github.com/carpsesdema/AvA_Troublemaker](https://github.com/carpsesdema/AvA_Troublemaker)
* **Developer (Kori):** [carpsesdema@gmail.com](mailto:carpsesdema@gmail.com) (For project-specific inquiries or feedback)