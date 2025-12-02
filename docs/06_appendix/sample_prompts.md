# Appendix: Sample Prompts

This section provides examples of the prompts used to instruct the LLM for various tasks. Effective prompt engineering is key to achieving accurate and structured outputs.

## B.1 Entity Extraction (Invoice)

This prompt is designed to guide the LLM to extract specific fields from an invoice and return them in a clean, valid JSON format.

**Role:** Expert financial document analyst  
**Task:** Extract structured data from this invoice  
**Format:** Return JSON only, no markdown

**Invoice Text:**  
`{text}`

**JSON Schema:**  
```json
{
  "invoice_number": "string",
  "amount": "number",
  "currency": "string",
  "issue_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "vendor": "string",
  "customer": "string",
  "line_items": [{"description": "string", "amount": "number"}]
}
```

**Rules:**
- If a field is missing, use `null`.
- Dates must be in `YYYY-MM-DD` ISO format.
- Currency must be the 3-letter ISO code (e.g., USD, EUR, ETB).

## B.2 RAG Question Answering

This is the prompt template used in the Retrieval-Augmented Generation (RAG) pipeline. It strictly instructs the model to answer a user's question based *only* on the retrieved context.

**Prompt:**
```
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
---
{context}
---

Question: {question}

Answer:
```
This "grounding" prompt is critical for preventing the LLM from hallucinating or using its general knowledge, thereby improving the faithfulness and reliability of the answers.

## B.3 Document Summarization

This prompt guides the model to produce a concise summary and a list of key, scannable points.

**Role:** You are a helpful assistant who specializes in summarizing financial documents for busy executives.

**Task:** Create a brief summary and a list of 3-4 key points from the following document text.

**Text:**
`{text}`

**Output Format:**

**Summary:**
<A concise, one-paragraph summary of the document's purpose and main content.>

**Key Points:**
- <Key point 1>
- <Key point 2>
- <Key point 3>
```
