# LinkedIn Post Draft

---

Your RAG pipeline is probably overkill.

I spent 6 months optimizing a RAG pipeline for ESG data extraction. Query rewriting, reranking, hybrid search -- every trick in the book. We got from 60% to 70% accuracy.

Then we put the whole document in the context window: 85%.

This led me down a research rabbit hole that became a PyData Amsterdam 2025 talk. Here are the 5 things I learned:

1. The 100k token threshold is real. Chroma's Context Rot research showed that performance degrades sharply after that, especially with distractors in the context.

2. Reranking is dead for modern LLMs. It improves retrieval metrics (MRR, recall) but has zero effect on actual answer quality. Modern models handle noisy context just fine.

3. You need way more chunks than you think. The standard k=5 or k=10 isn't enough. Performance saturated around k=50 in our tests.

4. Speed is comparable. For documents under 100k tokens, full context is about as fast as RAG.

5. Long context Q&A is still unsolved. Even at 113k tokens, models struggle with distractors, dissimilar phrasing, and conversational context.

The practical takeaway? If your domain fits under 100k tokens, try ditching RAG entirely. The simplicity gain alone is worth it.

Full write-up with all the experiment details on my blog (link in comments).

The context window quality findings lean heavily on Chroma's Context Rot article -- essential reading if you work with LLMs. The speed and reranking experiments were my own, made possible by orq.ai's unified LLM API access.

#PyData #RAG #LLM #MachineLearning #NLP #AI
