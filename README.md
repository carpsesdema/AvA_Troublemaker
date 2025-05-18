# AvA: Advanced Versatile Assistant
Your intelligent AI desktop partner, built on a flexible platform to supercharge complex tasks and workflows.

---

<p align="center">
  <img src="assets/AvA_Responding.gif" alt="AvA Demo - AI Assistant Responding" width="700">
</p>

---

## What is AvA?

In a world rapidly adopting Large Language Models (LLMs), effectively harnessing their power for complex, multi-step tasks directly on your desktop can be challenging. Managing different AI models, providing them with the right context from your local files, and orchestrating them for sophisticated workflows often requires significant boilerplate and specialized knowledge.

**AvA (Advanced Versatile Assistant)** is an intelligent desktop application designed by a solo developer to bridge this gap. It provides a rich, interactive environment where you can:

*   **Integrate and chat with local LLMs** (via Ollama, supporting models like CodeLlama and Llama3) for privacy and offline capability.
*   **Connect to powerful cloud-based models** like Google's Gemini API and OpenAI's GPT models.
*   **Leverage an advanced Retrieval Augmented Generation (RAG) system** that understands your local codebases, PDFs, and DOCX files, providing deep contextual awareness to the AI.
*   **Execute innovative multi-step AI workflows**, such as the built-in system for bootstrapping new applications, where a "Planner AI" outlines the project structure and a "Specialist Generator AI" creates the code for multiple files.

At its core, AvA is more than just a collection of features; it's built as an **extensible AI platform**. The initial release focuses on supercharging developer workflows, but its underlying architecture—which combines a general-purpose conversational "Planner" AI with slots for specialized "Task" AIs and a robust RAG system—is designed for versatility. This opens the door for AvA to be adapted for a wide range of sophisticated AI-powered assistance across various domains in the future.

## The "Aha!" Moment: How AvA's Core Architecture Came To Be

As a developer, I often found myself needing to generate substantial amounts of code or understand large, existing codebases. While powerful cloud-based LLMs like Gemini and GPT are excellent for many tasks, direct API calls for massive code generation can become expensive very quickly.

My initial workflow involved:

1.  **Building a Knowledge Base:** I'd upload extensive Python code (and other documents) into a local RAG (Retrieval Augmented Generation) system. This gave a local language model (initially, a coding-specific Llama model via Ollama) a solid foundation to draw upon.
2.  **Interacting with the Code Model:** I then built an interface to interact with this local code model, leveraging the RAG for context.

However, a challenge emerged: effectively communicating complex needs or nuanced instructions directly to a specialized code model like CodeLlama isn't always straightforward. These models are fantastic at generation but might not excel at broader understanding or conversational dialogue.

**This led to the core innovation in AvA: introducing a general-purpose LLM (like Gemini or a capable GPT/Llama3 chat model) to act as an intelligent "translator" and "planner."**

This "Planner AI":
*   Understands my natural language requests.
*   Can break down complex tasks (like planning a new multi-file application).
*   Interacts with the RAG system for deep context.
*   Then, intelligently prompts the specialized "Generator AI" (e.g., a local CodeLlama) to produce the precise code needed for each file.

This hybrid approach allows AvA to combine the conversational strengths and planning capabilities of general LLMs with the cost-effective, specialized (and often private) code generation power of local models. It's about using the right AI for the right part of the job, all orchestrated seamlessly through a user-friendly desktop interface. This architecture also naturally paved the way for AvA to be more than just a coding tool, but a versatile platform for various AI-driven tasks.

## Key Features

AvA brings a suite of powerful AI capabilities to your desktop, designed and built to enhance complex workflows:

*   **Hybrid LLM Integration:**
    *   **Local LLMs via Ollama:** Run models like CodeLlama, Llama3, and others directly on your machine for privacy, offline access, and cost-effective generation.
    *   **Cloud API Support:** Seamlessly connect to Google's Gemini API and OpenAI's GPT API to leverage their powerful models for advanced tasks.
*   **Advanced RAG (Retrieval Augmented Generation):**
    *   Chat with your local codebase (Python files and more).
    *   Index and query PDF and DOCX documents.
    *   Utilizes FAISS for efficient vector storage/search and Langchain for orchestration.
*   **Multi-File Application Bootstrapping:**
    *   Go beyond single-file generation with an AI-driven workflow to create new applications.
    *   A "Planner AI" (e.g., Gemini or GPT) strategizes the file structure and purpose for a new project.
    *   A "Generator AI" (e.g., local CodeLlama) implements the code for each planned file.
    *   View and manage generated code in the integrated Code Viewer.
*   **Extensible Platform Architecture:** AvA is built with modularity in mind. Its core design (Planner AI + Specialist AI + RAG) allows users to customize the AI setup for their specific needs and can be extended to support various AI models and tasks beyond the initial coding focus. The current model integrations (Ollama, Gemini, OpenAI) serve as a starting point.
*   **User-Focused Desktop Experience:**
    *   Developed with Python and PyQt6 for a responsive and native desktop feel.
    *   Project-based organization to manage different contexts and knowledge bases.
    *   Dedicated Code Viewer for reviewing AI-generated files.

## Current Status

AvA is currently in Alpha and is the result of a solo development effort. Your feedback and bug reports are highly valued and will directly contribute to its improvement!
*Note: Modifying existing codebases (beyond bootstrapping new ones) is a feature currently under refinement.*

## Getting Started (Alpha)

AvA is currently in Alpha. These instructions guide you on how to run it from the source code.

### Prerequisites

*   **Python:** Version 3.11+ is recommended (AvA was developed using Python 3.13).
*   **Git:** To clone the repository.
*   **Ollama (Recommended for Local LLMs):**
    *   If you wish to use local models like CodeLlama or Llama3, ensure Ollama is installed and running. Download it from [ollama.com](https://ollama.com/).
    *   **Important:** After installing Ollama, you must pull the specific models you intend to use via the Ollama command line. For example:
        ```bash
        ollama pull codellama
        ollama pull llama3
        ollama pull llava # For multimodal if you plan to use it
        ```
        AvA will indicate if a selected Ollama model is not found locally.

### Installation & Running

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/carpsesdema/AvA_Troublemaker.git
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

4.  **API Keys (Optional but Recommended for Cloud Models):**
    *   **Google Gemini:**
        *   To use Google Gemini models, you need an API key.
        *   The application looks for an environment variable named `GEMINI_API_KEY`.
    *   **OpenAI GPT:**
        *   To use OpenAI GPT models, you need an API key.
        *   The application looks for an environment variable named `OPENAI_API_KEY`.
    *   **Setting API Keys:**
        *   You can set these in your terminal session before running AvA:
            *   Linux/macOS:
                ```bash
                export GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
                export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
                ```
            *   Windows (PowerShell):
                ```powershell
                $env:GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
                $env:OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
                ```
        *   Alternatively, create a `.env` file in the project's root directory (the same folder as `main.py`) and add the lines:
            ```
            GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
            OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
            ```
    *   If API keys are not provided, features relying on those cloud models will be unavailable, but local Ollama models will still function if Ollama is properly set up.

5.  **Run AvA:**
    *   Ensure your virtual environment is activated.
    *   Execute the main script from the project's root directory:
        ```bash
        python main.py
        ```
    This should launch the AvA desktop application.

## Using AvA

AvA offers a flexible environment. Here’s a guide to its core functionalities:

### 1. Initial Setup & Configuration

*   **API Keys:** Ensure your API keys for Gemini and/or OpenAI are set up as described in "Getting Started" if you plan to use these models.
*   **Ollama Models:** If using Ollama, make sure you have pulled the desired models (e.g., `ollama pull llama3`, `ollama pull codellama`).
*   **Selecting LLMs (Left Panel):**
    *   **Chat LLM:** Use the "Chat LLM" dropdown to choose your primary conversational AI (e.g., a Gemini model, an Ollama Llama3 model, or a GPT model). This AI handles general chat, RAG interaction, and acts as the "Planner" for some complex tasks.
    *   **Specialized LLM:** Use the "Specialized LLM" dropdown to select a model for specific tasks, currently primarily for code generation (the "Generator AI"). Local Ollama models like CodeLlama are recommended here for privacy and cost-effectiveness.
    *   AvA will attempt to configure the selected models. Status messages will indicate success or any issues (e.g., missing API key, model not pulled for Ollama).
*   **Configure AI Persona (Optional):**
    *   Click "Configure AI Persona" in the left panel.
    *   Define a system prompt to guide the behavior and conversational style of your selected **Chat LLM**. This is useful for tailoring AvA's responses.
    *   *Note: The Specialized LLM (Generator AI) has its own specific system prompts managed internally by AvA for its tasks.*
*   **Temperature (Left Panel):**
    *   Adjust the "Temperature" slider (0.0 to 2.0) to control the creativity/randomness of the **Chat LLM's** responses. Lower values are more deterministic, higher values are more creative.

### 2. Working with Projects & Knowledge (RAG)

AvA uses **Projects** to isolate chat histories and knowledge bases.

*   **Create a New Project:**
    *   Click "Create New Project" in the left panel.
    *   Give your project a unique name (e.g., `my_python_api`, `research_topic_xyz`).
    *   The new project will appear in the "PROJECTS" list and become active.
*   **Select an Existing Project:**
    *   Click on a project name in the "PROJECTS" list to make it active. Its RAG knowledge and chat history will be loaded.
*   **Building Your Knowledge Base (RAG):**
    *   With a project selected, use the "Add File(s)" or "Add Folder" buttons under the "KNOWLEDGE FOR '[Project Name]'" section to upload relevant documents.
    *   **For Code Projects:** Upload your source code files or entire project folders. A powerful technique is to upload the `site-packages` directory from a Python virtual environment (`venv`) to give AvA deep knowledge about the libraries you're using.
    *   **For Documents:** Upload PDFs or DOCX files.
    *   AvA will process and index these files. The "View Project RAG" button allows you to inspect the indexed content.
*   **Global Knowledge:**
    *   Files uploaded via "Manage Global Knowledge" are added to a shared "AvA Global Knowledge" base, accessible across all projects.

### 3. Standard Chat & RAG Interaction

*   **Start a New Chat:**
    *   With your desired project selected (which opens its tab or creates a new one), click "New Chat" in the left panel. This clears the history in the active project's tab.
*   **Chatting with AvA:**
    *   Type your questions or requests into the input bar at the bottom of a chat tab.
    *   If the active project has a populated RAG knowledge base, AvA's **Chat LLM** will automatically use that context to provide more relevant and informed answers.
    *   You can ask for explanations, brainstorm ideas, get summaries of RAG content, etc.
    *   Attach images using the "+" button next to the input bar for multimodal interactions (if the selected Chat LLM supports it, like some Gemini or Llava models).

### 4. Advanced Workflow: Bootstrapping New Applications

AvA can help you generate the initial structure and code for a new multi-file application.

*   **How to Trigger:**
    *   In a chat tab (preferably within a new, empty project context you've created for this new application), describe the application you want to build. Be specific about the language, framework, and key features.
    *   Use clear "bootstrap" or "create new app" phrasing, for example:
        *   `"Bootstrap a new Python Flask application for a to-do list."`
        *   `"Create a new project: a simple web server in Go that serves static files from a 'public' directory."`
        *   `"Generate a new Python script that takes a CSV file as input, processes column X, and outputs a summary to a text file."`
*   **The Process:**
    1.  **Planning Phase:** Your request is sent to the **Planner AI** (your selected Chat LLM). The Planner will:
        *   Outline a list of files needed for the new application.
        *   Create a high-level "proto-specification" describing the purpose of each file and the overall project.
        *   Generate specific, detailed instructions for the **Generator AI** for each file.
        *   AvA will display status messages like "[System: Planner AI is outlining...]" and "[System: Planner AI is generating Coder instructions...]".
    2.  **Code Generation Phase:** The instructions for each file are sent one by one to the **Specialized LLM** (your selected Generator AI, e.g., CodeLlama).
        *   The Generator AI produces the code for each file.
        *   AvA displays status messages like "[System: Processing file X/Y: Generating code for `filename.ext`...]".
*   **Viewing Generated Code:**
    *   As files are generated, they will appear in the **Code Viewer**. Click "View Code Blocks" in the left panel to open it.
    *   You can select files in the Code Viewer's list to see their content.
*   **Completion & Next Steps:**
    *   Once all files are attempted, AvA will provide a summary message in the chat indicating success or any failures.
    *   At this point, you can review the generated code in the Code Viewer.
    *   *(The ability to iteratively refine the generated code through further chat interaction is a planned feature and currently under development.)* For now, you would typically copy the code from the Code Viewer into your local development environment.

### 5. Tips for Effective Use

*   **Be Specific with Prompts:** The more detail you provide in your requests, especially for code generation or complex RAG queries, the better the results will be.
*   **Curate Your RAG:** For project-specific RAG, only upload files directly relevant to that project to avoid noise. Use the "Global Knowledge" for generally useful documents.
*   **Choose the Right LLMs:**
    *   For general chat and planning, a capable model like Gemini Pro, GPT-4, or a large Llama3 variant is good.
    *   For code generation, specialized models like CodeLlama (run locally via Ollama) are excellent and can be more cost-effective or private.
*   **Experiment with Temperature:** Adjust the temperature for your Chat LLM to find the right balance between factualness and creativity for your needs.
*   **Use Projects:** Leverage projects to keep your contexts and chat histories organized. This is crucial for effective RAG.

## Technologies Used

AvA is built with a powerful stack of Python libraries and tools:

*   **Core Application:**
    *   Python 3.11+
    *   PyQt6 for the native desktop graphical user interface.
    *   qasync for integrating asyncio with PyQt6.
*   **LLM Integration:**
    *   `google-generativeai` for interacting with Google Gemini API models.
    *   `openai` for OpenAI GPT API models.
    *   `ollama` client library for seamless communication with local Ollama instances.
*   **RAG System:**
    *   `faiss-cpu` for efficient similarity search in vector stores.
    *   `sentence-transformers` for generating text embeddings.
    *   `langchain-text-splitters` (specifically `RecursiveCharacterTextSplitter` and `PythonCodeTextSplitter`) for intelligent document chunking.
    *   `PyPDF2` for PDF document parsing.
    *   `python-docx` for DOCX document parsing.
*   **UI Enhancements & Utilities:**
    *   `Markdown` for rendering rich text in chat.
    *   `Pillow` for image handling.
    *   `qtawesome` for easier icon integration in the UI.
    *   `python-dotenv` for managing environment variables.
*   **Development & Structure:**
    *   Organized into core services, backend adapters, UI components, and utility modules.
    *   Features extensive logging for diagnostics and debugging.

## How to Contribute

AvA is currently an alpha release developed by a solo developer. Feedback, bug reports, and feature suggestions are highly welcome and invaluable at this stage!

*   **Reporting Issues:** If you encounter any bugs or unexpected behavior, please open an issue on the [GitHub Issues page](https://github.com/carpsesdema/AvA_Troublemaker/issues).
*   **Feature Requests:** Have an idea that could make AvA even better? Feel free to submit it as a feature request on the Issues page.

As the project matures, guidelines for code contributions may be established.

## Support AvA's Development

AvA is a passion project. If you find it useful or believe in its vision, please consider supporting its continued development:

*   [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/snowballKori)

Your support helps cover development costs and allows for more time to be dedicated to improving AvA.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contact & Links

*   **Website / Landing Page:** [snowballannotation.com](http.snowballannotation.com)
*   **GitHub Repository:** [https://github.com/carpsesdema/AvA_Troublemaker](https://github.com/carpsesdema/AvA_Troublemaker)
*   **Developer (Kori):** [carpsesdema@gmail.com](mailto:carpsesdema@gmail.com) (For project-specific inquiries or feedback)