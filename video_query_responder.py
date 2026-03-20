from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class VideoAssistant:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key="AIzaSyD_djBtv_Bj9z869t-n_fA9H7dboN99Zvw"
        )

    def answer_question(self, context, question):
        prompt = f"""
        Context: {context}
        Question: {question}
        Answer clearly for a student.
        """

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content

# Test context
test_context = """
- Visual: Gameplay showing character selection screen with multiple avatars
- Text on screen: Level 1: Training Grounds | Choose Your Character
- Instructor just said: "Now we need to choose our starting character. Each has different abilities."
- Topic: Gaming
- Difficulty: Intermediate
"""

test_question = "What character should I choose?"
assistant = VideoAssistant()

# Run it
response = assistant.answer_question(test_context, test_question)
print(response)