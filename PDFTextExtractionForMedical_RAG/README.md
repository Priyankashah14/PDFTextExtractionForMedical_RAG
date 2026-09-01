This is the simple RAG pipeline which reads the medical PDF and answer the question. Pre-requisites: All the modules are installed. Refer: pyproject.toml Run: PDF_Text_Extraction.py --> This will create ChromaDB Run: Question_to_RAG.py --> ask question to the model.

you can test this by asking questions like:

What is mitral valve regurgitation?
What is LV volume?
Give me equation for volume?
###################################################
Enhancement to PDFTextExtractionRAG
Check ConversationChatBot_PDF folder.

This script implements a conversational chatbot that interacts with PDF content using Groq API and Langchain libraries. It allows users to upload PDF files, process them, and ask questions based on the extracted information.
you can execute the code by typing this command in terminal. stremlit run "Directory\name of the script.python" 
