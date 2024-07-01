
# from langchain.llms.base import LLM
# from typing import List, Dict, Any
# class TransformersLLM(LLM):
#     def __init__(self, pipe):
#         super().__init__()
#         object.__setattr__(self, 'pipe', pipe)

#     @property
#     def _llm_type(self) -> str:
#         return "transformers"

#     def _call(self, question: str, context: str) -> str:
#         result = self.pipe(question=question, context=context)
#         return result['answer']

#     def _generate(self, prompts: List[str], contexts: List[str], **kwargs) -> List[Dict[str, Any]]:
#         return [{"text": self._call(prompt, context)} for prompt, context in zip(prompts, contexts)]


# def generate_summary_statement(chunks: List[str], question: str, llm: TransformersLLM) -> str:
#     print("Lookin")
#     combined_text = " ".join(chunks)
#     prompt = f"Summarize the following information to answer the question: {question}\n\n{combined_text}"
#     max_length = 512  # Maximum length for the model
#     chunk_size = max_length - len(question) - 10  # Ensure enough space for the question and other tokens

#     # Split the context into manageable chunks
#     split_contexts = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
    
#     summaries = []
#     for context in split_contexts:
#         if len(context.strip()) == 0:
#             continue
#         summary = llm._call(question, context)
#         summaries.append(summary)
    
#     return " ".join(summaries)

# model.py
import os
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

class QuestionAnsweringAgent:
    def __init__(self, model_name='deepset/roberta-base-squad2', vector_index_path='vector_index.pkl'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        self.vector_index_path = vector_index_path
        self.vector_index = self.load_vector_index()

    def load_vector_index(self):
        if os.path.exists(self.vector_index_path):
            with open(self.vector_index_path, 'rb') as f:
                vector_index = pickle.load(f)
            return vector_index
        else:
            raise FileNotFoundError(f"Vector index file not found at {self.vector_index_path}")

    def get_relevant_docs(self, query):
        retriever = self.vector_index.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in relevant_docs]

    def generate_answer(self, query):
        relevant_docs = self.get_relevant_docs(query)
        context = " ".join(relevant_docs)
        qa_input = {
            'question': query,
            'context': context
        }
        result = self.qa_pipeline(qa_input)
        if result['score'] < 0.1:  # Threshold can be adjusted
            return "No relevant information found for your query."
        return result['answer']

# Example usage
if __name__ == "__main__":
    agent = QuestionAnsweringAgent()
    query = "What is the impact of climate change on agriculture?"
    answer = agent.generate_answer(query)
    print(f"Answer: {answer}")
