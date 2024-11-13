import os
import re
import subprocess
import tempfile
from pathlib import Path

import docker
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Code Interpreter")
st.header("Code Interpreter")

help_text = """
⚡️ Environment to execute the generated code.

⚠️ Be careful running arbitrary code on your local machine
- None: Do not run the generated code.
- docker: Run code inside a docker container.
- local: Run code directly on your local machine.
"""

execution_env = st.selectbox(
    "Select environment for Code Execution: ",
    ("None", "docker", "local"),
    help=help_text,
)

user_prompt = st.text_area("**Prompt:**", placeholder="Enter your prompt...")
run = st.button("Run")

local_execution_prompt = """
You are an intelligent AI agent designed to generate accurate python code.
Here are your STRICT instructions:
- If the question does not require writing code, provide a clear and concise answer without generating any code.
- Whatever question is asked to you generate just the python code based on that and it's important that you just generate the code, no explanation of the code before or after.
- Think step by step how to solve the problem before you write the code.
- If the code involves doing system level tasks, it should have the code to identify the platform, the $HOME directory to make sure the code execution is successful.
- Code should be inside of the ```python ``` block.
- Make sure all the imports are always there to perform the task.
- At the end of the generated code, always include a line to run the generated function.
- The code should print output in a human-readable and understandable format.
- If you are generating charts, graphs or anything visual, convert them to image and save it to the /tmp location and return as well as print just the name of the image file path.
"""

docker_execution_prompt = """
You are an intelligent AI agent designed to generate accurate python code.
Here are your STRICT instructions:
- If the question does not require writing code, provide a clear and concise answer without generating any code.
- Whatever question is asked to you generate just the python code based on that and it's important that you just generate the code, no explanation of the code before or after.
- Think step by step how to solve the problem before you write the code.
- If the code involves doing system level tasks, it should have the code to identify the platform, the $HOME directory to make sure the code execution is successful.
- Code should be inside of the ```python ``` block.
- Make sure all the imports are always there to perform the task.
- At the end of the generated code, always include a line to run the generated function.
- The code should print output in a human-readable and understandable format.
- If you are generating charts, graphs or anything visual, convert them to image and save it to the /app location and return as well as print just the name of the image file without path.
"""

if execution_env == "docker":
    messages = [SystemMessage(docker_execution_prompt)]
else:
    messages = [SystemMessage(local_execution_prompt)]


def get_code_group(llm_response: str) -> str | bool:
    """Extracts Python code from a markdown-formatted LLM response.

    Args:
        llm_response: A string containing the full response from the LLM, potentially
            containing Python code blocks.

    Returns:
        str: The extracted Python code if found.
        bool: False if no Python code block is found.
    """
    code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    print(">>> Code Match: ", code_match)
    if not code_match:
        print(">>> No Python code found in the response.")
        return False

    return code_match.group(1)


def execute_local(temp_file_path: str) -> str:
    """Executes a Python script locally and returns its output.

    Runs the given Python script file using python3.10 interpreter in a subprocess
    with a x second timeout. Captures and returns stdout/stderr.

    Args:
        temp_file_path: Path to the temporary Python script file to execute.

    Returns:
        str: The stdout output if execution was successful, stderr if there were errors,
            or a timeout message if execution exceeded 10 seconds.

    Raises:
        subprocess.TimeoutExpired: If script execution takes longer than 10 seconds.

    Note:
        The temporary file is deleted after execution regardless of success/failure.
    """
    try:
        result = subprocess.run(
            ["python3.10", temp_file_path], capture_output=True, text=True, timeout=10
        )
        print(">>> Running code completed!")
        return result.stdout if result.returncode == 0 else result.stderr
    except subprocess.TimeoutExpired:
        return ">>> Execution timed out."
    finally:
        os.unlink(temp_file_path)


def execute_docker(temp_file_path: str) -> str:
    """Executes a Python script in a Docker container and returns its output.

    Runs the given Python script file in a Docker container using the code-interpreter-demo
    image. Mounts the temporary script directory into the container and executes the script.

    Args:
        temp_file_path: Path to the temporary Python script file to execute.

    Returns:
        str: The stdout output from the container if execution was successful,
            or an error message if container execution failed.

    Raises:
        Exception: If there are any errors running the Docker container.

    Note:
        The temporary file is deleted after execution regardless of success/failure.
        The container is automatically removed after execution.
    """
    client = docker.from_env()
    try:
        container = client.containers.run(
            "code-interpreter-demo:latest",
            command=f"python /app/{Path(temp_file_path).name}",
            volumes={os.path.dirname(temp_file_path): {"bind": "/app", "mode": "rw"}},
            remove=True,
        )
        return container.decode("utf-8")
    except Exception as e:
        return f"Failed running the container: {str(e)}"
    finally:
        os.unlink(temp_file_path)


if run:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    messages += [HumanMessage(user_prompt)]
    ai_message = llm.stream(messages)
    output = st.write_stream(ai_message)

    code = get_code_group(output)
    if output and code:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
            print(f">>> Temp file path is : {temp_file_path}")

        executed_result = None
        if execution_env == "local":
            print(">>> Executing in Local Machine!")
            executed_result = execute_local(temp_file_path)
        elif execution_env == "docker":
            print(">>> Executing in Docker!")
            executed_result = execute_docker(temp_file_path)

        if executed_result and any(ext in executed_result for ext in (".png", ".jpg")):
            if execution_env == "local":
                st.image(executed_result.strip())
            else:
                st.image(f"{os.path.dirname(temp_file_path)}/{executed_result.strip()}")
        else:
            st.write(executed_result)
