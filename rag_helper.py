from ingest import build_index, load_faq_data

from openai import OpenAI
import os
import sqlite3

from db_search import SQLiteFTS

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = '''
QUESTION: 
{question}

CONTEXT:
{context}
'''

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='openai/gpt-oss-20b'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, question, course="llm-zoomcamp"):
        boost_dict = {'question': 2.0, 'section': 0.5}
        filter_dict = {"course": course}

        return self.index.search(
            question, 
            filter_dict=filter_dict,
            num_results=5,
            boost_dict=boost_dict
            )

    def build_context(self, search_result):
        lines = []

        for doc in search_result:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, question, search_result):
        context = self.build_context(search_result)
        return self.prompt_template.format(question=question, context=context)

    def llm(self, prompt):
        message_history = [
            {"role": 'developer', 'content': self.instructions},
            {"role": 'user', 'content': prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=message_history
        )
        return response.output_text

    def rag(self, query):
        # 1. Retrieve the most relevant documents
        search_result = self.search(query)

        # 2. Generate the prompt with the most relevant documents
        prompt = self.build_prompt(query, search_result)
        
        # 3. Augment the context with the llm
        answer = self.llm(prompt)

        return answer

class RAGBaseSQLite(RAGBase):

    def __init__(self, db_path):
        super().__init__(
            index=SQLiteFTS(db_path),
            llm_client=OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            ),
            course='llm-zoomcamp', model='openai/gpt-oss-20b'
        )

    def search(self, question, course="llm-zoomcamp", num_results=5):
        return self.index.search(question, course=course, num_results=num_results)