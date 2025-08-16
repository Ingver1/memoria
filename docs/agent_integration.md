# LLM-Agent Integration Tutorial

This tutorial walks through wiring an LLM agent to the memoria service so the
agent can persist and recall memories during conversations.

## Prerequisites
- Running memoria API (see [deployment guide](deployment.md)).
- OpenAI or other embedding provider configured.

## Steps
1. **Install SDK**
   ```bash
   pip install memoria-client
   ```
2. **Initialize the client**
   ```python
   from memoria.memory_system.api.client import MemoryClient
   memory = MemoryClient("http://localhost:8000")
   ```
3. **Store observations**
   After the agent processes user input, add key facts:
   ```python
   memory.add(text=observation, metadata={"user": session_id})
   ```
4. **Route and retrieve context**
   Before generating the next response, decide whether to search and which
   channels to target:
   ```python
   from memoria.memory_system.rag_router import Router

   router = Router()
   decision = router.decide(user_message)
   if decision.use_retrieval:
       context = memory.search(query=user_message, limit=5)
   ```
   The returned ``RouterDecision`` details the chosen channels and whether the
   query was treated as global or personal.
5. **Augment prompts**
   Combine retrieved snippets with the user message when calling the LLM.

## Result
The agent accumulates long-term knowledge while keeping prompts concise,
enabling richer interactions over time.
