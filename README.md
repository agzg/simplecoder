# SimpleCoder
**Collaboration notice:** Ali Azam, Promita Rahee Sikder

**SimpleCoder**, is a CLI coding agent that can help you write code, navigate codebases, and complete various software engineering tasks. This agent uses several useful concepts from the modern AI Agent development stack: tool use, Retrieval-Augmented Generation (RAG), context management, and task planning, etc.

## Overview

SimpleCoder is a ReAct-style agent that combines tool use, semantic code search, context management, and task planning.

### Requirements
Since `uv` is doubtlessly the superior `pip`, use: 
```bash
uv sync
```
to install all of the requirements. We used the Dartmouth Chat API for LiteLLM and embeddings, so make sure you have:
```bash
export DARTMOUTH_CHAT_API_KEY="your-key-here" # You can also switch to a different provider, in case you prefer
```
or it is set up in an .env file.

### Usage
For the basic implementation, you can do things like:
```bash
# basic usage
uv run simplecoder "create a hello.py file"

# with RAG
uv run simplecoder --use-rag "what does the Agent class do?"

# with planning
uv run simplecoder --use-planning "create a web server with routes for home and about"

# interactive mode
uv run simplecoder --interactive
# or
uv run simplecoder # [flags] but don't add a string task here

# verbose mode (would reccomend enabling)
uv run simplecoder --verbose

# help options
uv run simplecoder --help
```

Reflections (adds observations from failed strategies or user corrections):
```bash
uv run simplecoder --use-reflection
```

Dangerous mode (run_shell: agent can run pytest, python scripts; you approve each command):
```bash
uv run simplecoder --dangerous "implement and test a calculator module"
```

### Extra features
* Reflections on strategies, failed to tool calls, and explicit user feedback (with `--use-reflections`) inspired by "Generative Agents" by Park, et al. 2023. Uses an LLM to assign an importance score to each action-observation pair. Also helps detects cycles (see below).
* `use_llm` tool call which allows it to call query an LLM to understand code, research on topics, and even delegate subtasks in larger code projects.
* Detects cycles in tool calls (prevent infinitely un/doing same action), and picks a new strategy.
* Prevents unsafe code and execution. With `--dangerous`, run_shell allows running pytest/python; commands are safety-checked and you must approve (y/n) each run.
* Smarter summarization - tries to preserve filenames, tool call details, etc.
* More robust instructions to prevent hijacking agent behavior or internal representations (like reflections).
* Continues context when max_iterations is reached in interactive mode.

## Demo

1. **Make the game**:
Designed the adventure game (in `game.py`) after researching Harry Potter and the Chamber of Secrets, wrote the code, debugged it, and ran the game twice losing and winning. We then show you some ways in which it tries to prevent hijacking and preserves context from the conversation history.

https://github.com/user-attachments/assets/eb1a8b6b-1ec0-4857-8a58-5d9a1b463644

2. **Add Voldemort and fix errors**
We ran out of credits before adding Voldemort, so I use another account's credit by switching out the API key in the .env file and allow it to read the file without the context from the conversation history earlier, and then adds the Voldemort character. We intentionally left some syntax errors, and it identifies them via reflections and corrects them before proceeding. Then it plays the game once and is lucky enough not to be killed by Voldemort.

https://github.com/user-attachments/assets/f5c4f572-29b5-4cbc-94e5-be1aa75709d9
