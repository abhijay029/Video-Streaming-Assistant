from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import numpy as np
from helper.system_prompts import SystemPrompt

load_dotenv(".env")

class FrameInterpreter:

    system_prompt = SystemPrompt.get_frame_interpretor_prompt()
    
    def __init__(self):

        self.agent = create_agent(
            model = "google_genai:gemini-2.5-flash",
            system_prompt= self.system_prompt,
            checkpointer = InMemorySaver()
        )

        self.config = {'configurable': {"thread_id": "1"}}

    def get_question(self, frames: list, transcript: str):
        
        if type(frames) != np.ndarray:
            frames = np.array(frames)

        if frames.ndim != 1:
            raise ValueError("frames is not a 1-dimensional array.")


        frames_message = HumanMessage(content = [{"type": "text", "text": "Analyze these video frames and the transcript carefully and return structured JSON output."},
                                                 *[{"type":"image", "base64": frame, "mime_type": "image/jpg"} for frame in frames],
                                                 {"type": "text", "text": f"Transcript:\n{transcript}"}])  

        return frames_message


    def interpret_frames(self, frames: list, transcript: str):

        question = self.get_question(frames = frames, transcript = transcript)
        
        response = self.agent.invoke(
            {
            "messages": [question]
            },
            config = self.config
        )

        interpretation = response["messages"][-1].content

        return interpretation
