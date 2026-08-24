This is the simple RAG pipeline which reads the medical PDF and answer the question.
Pre-requisites: All the modules are installed. Refer: pyproject.toml
Run: PDF_Text_Extraction.py --> This will create ChromaDB
Run: Question_to_RAG.py --> ask question to the model.

you can test this by asking questions like:
  1. What is mitral valve regurgitation?
  2. What is LV volume?
  3. Give me equation for volume?
