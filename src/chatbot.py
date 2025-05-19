import os
from .ai_function import medicalbot_ai
from .database import search_db
from loguru import logger


def medical_chatbot(user_query: str)-> str:
    logger.info("Searching for similar Docs in the Vec DB")
    doc_list = search_db(user_query=user_query)
    logger.info("Calling the medicalbot ai to get answer")
    answer = medicalbot_ai(user_query=user_query, doc_list=doc_list)
    logger.info("Returning the Final Answer")
    return answer


# if __name__ == "__main__":
#     question = "What is a fungus?"
#     answer = medical_chatbot(user_query=question)
#     print(answer)