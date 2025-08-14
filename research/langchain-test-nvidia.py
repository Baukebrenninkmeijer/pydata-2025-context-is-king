from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct")

# for chunk in llm.stream([{"role": "user", "content": "What seems to be the problem?"}]):
#     print(chunk.content, end="")

# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# client = NVIDIAEmbeddings(
#     model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
#     truncate="NONE",
# )

# embedding = client.embed_query("What is the capital of France?")
# print(len(embedding))

prompts = [
    "What is AI?",
    "Define quantum computing.",
    "Explain gravity.",
    "What is the capital of Japan?",
    "Summarize the internet.",
    "Who wrote Hamlet?",
    "Describe a black hole.",
    "What is photosynthesis?",
    "List three prime numbers.",
    "Translate 'hello' to French.",
    "What is blockchain?",
    "Name a famous painting.",
    "Explain relativity.",
    "What is Python?",
    "Give a fun fact.",
    "What is the speed of light?",
    "Who is Ada Lovelace?",
    "Describe a rainbow.",
    "What is a neuron?",
    "What causes tides?",
]
out = llm.batch(inputs=prompts + prompts + prompts)
print(out)
