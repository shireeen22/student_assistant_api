from fastapi import FastAPI
from llm import (create_vector_store,load_vector_store,explain_text,generate_notes,quiz_generator,ask_question)
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return{
        "message": "Student assistant API running successfully..."
    }


class ChapterRequest(BaseModel):
    text: str

class QuestionRequest(BaseModel):
    question: str


@app.post("/process-chapter")
def process_chapter(request: ChapterRequest):

    try:

        create_vector_store(request.text)

        return {
            "success": True,
            "message": "Chapter processed successfully"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
#************************************Explaination Route**************************

@app.get("/explain")
def explain():

    try:

        vector_store = load_vector_store()

        response = explain_text(vector_store)

        return {
            "success": True,
            "explanation": response
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
#***************************************Generate Notes route***********************************
@app.get("/notes")
def notes():
    try:
        vector_store = load_vector_store()
        response = generate_notes(vector_store)
        return {
            'success': True,
            'notes': response
        }
    except Exception as e:
        return{
            'success': False,
            'error': str(e)
        }
    
#***************************************quizz generator route****************
@app.get("/quiz")
def quiz():
    try:
        vector_store = load_vector_store()
        response = quiz_generator(vector_store)
        return {
            'success':True,
            'quiz':response
        }
    except Exception as e:
        return{
            'success':False,
            'error': str(e)
        }

#**************************************ask question route************************
@app.post("/ask")
def ask_ai(request:QuestionRequest):
    try:
        vector_store = load_vector_store()
        response = ask_question(vector_store,request.question)
        return {
            'success':True,
            'question': request.question,
            'answer': response
        }
    except Exception as e:
        return{
            'success': False,
            'error': str(e)
        }

@app.get("/ping")
def ping():
    return {"status": "alive"}