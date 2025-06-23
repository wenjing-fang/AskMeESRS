import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import csv_to_documents, create_vector_store, load_chatmodel, Retriever_builder, chat_input_ensembler, chatbot
from tqdm import tqdm

def evaluate_chatbot_questions(questions_list):

    documents = csv_to_documents(csv_path='../source_text/scrapping_results_v2.csv')
    vector_store = create_vector_store(documents)

    chatmodel = load_chatmodel()

    retriever = Retriever_builder(vector_store)
    answers_list = []
    #use tqdm to show the progress
    for index, q in tqdm(enumerate(questions_list), total=len(questions_list), desc="Evaluating questions"):
        chat_input = chat_input_ensembler(q, retriever)
        try:
            answer = chatbot(chat_input, chatmodel)
            answers_list.append(answer)

        except Exception as e:
            answers_list.append(None)

    return answers_list


# Read the question csv
df_questions = pd.read_csv('../evaluation/evaluation_question/evaluation_question.csv')
df_questions = df_questions.dropna(subset=['Question'])

# Get the answer from the chatbot
questions_list = df_questions['Question'].tolist()
anwser_list = evaluate_chatbot_questions(questions_list)
df_answer = pd.DataFrame({'Question': questions_list, 'Answer': anwser_list})

# Save the answer to csv
os.makedirs('../evaluation/evaluation_answer', exist_ok=True)
df_answer.to_csv('../evaluation/evaluation_answer/evaluation_answer.csv', index=False)
