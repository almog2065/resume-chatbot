from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)

OUT_OF_TOKENS_MESSAGE = (
    "We are currently unable to process your request because your token "
    "allocation appears to be exhausted across all supported providers "
    "(OpenAI, Groq, Gemini, and DeepSeek). Please review your usage "
    "limits or update your billing details before trying again."
)


def get_external_llm_key():
    """
    Return the first available external LLM API key in priority order:
    1. OpenAI
    2. Groq
    3. Gemini
    4. DeepSeek

    It looks only at environment variables and does not make any network calls.

    Returns:
        (provider_name, api_key) tuple, where provider_name is one of
        'openai', 'groq', 'gemini', or 'deepseek'.

    Raises:
        RuntimeError: if none of the supported API keys are set.
    """
    provider_env_map = [
        ("openai", "OPENAI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
    ]

    for provider, env_var in provider_env_map:
        api_key = os.getenv(env_var)
        if api_key:
            return provider, api_key

    raise RuntimeError(OUT_OF_TOKENS_MESSAGE)


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.name = "Almog Ben Simon"
        reader = PdfReader("/Users/almogbensimon/Projects/agents/1_foundations/me/Almog Ben-Simon CV.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("/Users/almogbensimon/Projects/agents/1_foundations/me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = None
            last_error = None

            # Try each provider in priority order for this call.
            for provider, env_var, base_url, model in [
                ("openai", "OPENAI_API_KEY", None, "gpt-4o-mini"),
                ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
                ("gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-1.5-flash"),
                ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
            ]:
                api_key = os.getenv(env_var)
                if not api_key:
                    continue

                client_kwargs = {"api_key": api_key}
                if base_url is not None:
                    client_kwargs["base_url"] = base_url

                client = OpenAI(**client_kwargs)

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                    )
                    break
                except Exception as e:
                    last_error = e
                    continue

            if response is None:
                # No provider succeeded for this call.
                # Optionally log last_error somewhere if needed.
                return OUT_OF_TOKENS_MESSAGE

            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    