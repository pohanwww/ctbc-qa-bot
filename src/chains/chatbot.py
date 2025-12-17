"""
CTBC Chatbot Chain.

Implements the main chatbot logic using LangChain LCEL, combining:
- RAG retrieval for CTBC FAQ knowledge
- Conversation history management
- Prompt engineering for customer service behavior
- Multiple response modes for comparison
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.llm.loader import LLMWrapper
from src.rag.retriever import FAQRetriever

logger = logging.getLogger(__name__)


# System prompts
SYSTEM_PROMPT_BASE = """You are a helpful, professional, and friendly English-speaking customer service agent for CTBC Bank (中國信託商業銀行).

Your role is to assist customers with questions about banking products and services.

Guidelines:
- Be warm, patient, and helpful
- Provide clear, professional responses
- If you don't know something specific, acknowledge it honestly
- Keep responses concise but complete"""

SYSTEM_PROMPT_RAG = """You are a helpful, professional, and friendly English-speaking customer service agent for CTBC Bank (中國信託商業銀行).

Your role is to assist customers with questions about CTBC Bank's products, services, fees, and policies.

IMPORTANT GUIDELINES:

1. **Use FAQ Context**: Base your factual answers ONLY on the provided FAQ context below. Do not make up information about CTBC Bank's specific products, rates, fees, or policies.

2. **Language**: Always respond in clear, professional English. If the FAQ context is in Chinese, translate and explain it clearly in English.

3. **When Information is Missing**: If the FAQ context does not contain relevant information:
   - Acknowledge that you don't have that specific information
   - Suggest contacting CTBC Bank directly:
     - Customer Service Hotline: 0800-024-365 (Taiwan)
     - International: +886-2-2745-8080

4. **Safety**: Do NOT provide specific financial, legal, or tax advice.

5. **Tone**: Be warm, patient, and helpful.

---

FAQ CONTEXT:
{context}

---

Answer based ONLY on the FAQ context above."""


class CTBCChatbot:
    """
    Main chatbot class for CTBC customer service using LangChain LCEL.

    Supports three modes:
    - base: Pure LLM response (no RAG)
    - rag: LLM + RAG retrieval
    - finetuned_rag: Fine-tuned LLM + RAG retrieval
    """

    def __init__(
        self,
        llm: LLMWrapper,
        retriever: FAQRetriever,
        finetuned_llm: LLMWrapper | None = None,
        max_history_turns: int = 10,
    ):
        """
        Initialize the CTBC chatbot.

        Args:
            llm: The base language model wrapper
            retriever: The FAQ retriever instance
            finetuned_llm: Optional fine-tuned language model
            max_history_turns: Maximum conversation turns to maintain
        """
        self.llm = llm
        self.retriever = retriever
        self.finetuned_llm = finetuned_llm
        self.max_history_turns = max_history_turns

        # Conversation history
        self.conversation_history: list[HumanMessage | AIMessage] = []

        # Build chains using LCEL
        self._build_chains()

        logger.info("CTBC Chatbot initialized with LCEL chains")

    def _build_chains(self) -> None:
        """Build LangChain LCEL chains for different modes."""

        # Output parser
        output_parser = StrOutputParser()

        # ========== Base Chain (No RAG) ==========
        base_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_BASE),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Chain: prompt | llm | parser
        self.base_chain = (
            {
                "question": RunnablePassthrough(),
                "history": RunnableLambda(
                    lambda _: self.conversation_history[-self.max_history_turns * 2 :]
                ),
            }
            | base_prompt
            | RunnableLambda(lambda x: self._format_prompt_for_model(x))
            | self.llm.model
            | output_parser
            | RunnableLambda(self._postprocess_response)
        )

        # ========== RAG Chain ==========
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_RAG),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Chain: retrieve context | prompt | llm | parser
        self.rag_chain = (
            {
                "context": RunnableLambda(
                    lambda x: self.retriever.get_relevant_context(x["question"])
                ),
                "question": lambda x: x["question"],
                "history": RunnableLambda(
                    lambda _: self.conversation_history[-self.max_history_turns * 2 :]
                ),
            }
            | rag_prompt
            | RunnableLambda(lambda x: self._format_prompt_for_model(x))
            | self.llm.model
            | output_parser
            | RunnableLambda(self._postprocess_response)
        )

        # ========== Fine-tuned RAG Chain ==========
        if self.finetuned_llm:
            self.finetuned_rag_chain = (
                {
                    "context": RunnableLambda(
                        lambda x: self.retriever.get_relevant_context(x["question"])
                    ),
                    "question": lambda x: x["question"],
                    "history": RunnableLambda(
                        lambda _: self.conversation_history[-self.max_history_turns * 2 :]
                    ),
                }
                | rag_prompt
                | RunnableLambda(lambda x: self._format_prompt_for_model(x))
                | self.finetuned_llm.model
                | output_parser
                | RunnableLambda(self._postprocess_response)
            )
        else:
            self.finetuned_rag_chain = None

    def _format_prompt_for_model(self, prompt_value: Any) -> str:
        """
        Format ChatPromptTemplate output to string for the model.

        Args:
            prompt_value: The prompt value from ChatPromptTemplate

        Returns:
            Formatted prompt string for Qwen model
        """
        messages = prompt_value.to_messages()

        prompt_parts = []
        for msg in messages:
            if msg.type == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.type == "human":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.type == "ai":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    def _postprocess_response(self, response: str) -> str:
        """Post-process the LLM response."""
        response = response.strip()

        # Remove special tokens
        special_tokens = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"]
        for token in special_tokens:
            response = response.replace(token, "")

        return response.strip()

    def chat_base(self, question: str) -> str:
        """
        Get response using base LLM only (no RAG).

        Args:
            question: User's question

        Returns:
            LLM response without RAG context
        """
        try:
            response = self.base_chain.invoke(question)
            return response
        except Exception as e:
            logger.error(f"Error in base chain: {e}")
            return f"Error: {str(e)}"

    def chat_rag(self, question: str) -> str:
        """
        Get response using LLM + RAG.

        Args:
            question: User's question

        Returns:
            LLM response with RAG context
        """
        try:
            response = self.rag_chain.invoke({"question": question})
            return response
        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            return f"Error: {str(e)}"

    def chat_finetuned_rag(self, question: str) -> str:
        """
        Get response using fine-tuned LLM + RAG.

        Args:
            question: User's question

        Returns:
            Fine-tuned LLM response with RAG context
        """
        if not self.finetuned_rag_chain:
            return "[Fine-tuned model not loaded]"

        try:
            response = self.finetuned_rag_chain.invoke({"question": question})
            return response
        except Exception as e:
            logger.error(f"Error in fine-tuned RAG chain: {e}")
            return f"Error: {str(e)}"

    def chat_all(self, question: str) -> dict[str, str]:
        """
        Get responses from all three modes for comparison.

        Args:
            question: User's question

        Returns:
            Dictionary with responses from each mode
        """
        # Get responses with individual error handling
        responses = {}

        try:
            responses["base"] = self.chat_base(question)
        except Exception as e:
            logger.error(f"Error getting base response: {e}")
            responses["base"] = f"Error: {str(e)}"

        try:
            responses["rag"] = self.chat_rag(question)
        except Exception as e:
            logger.error(f"Error getting RAG response: {e}")
            responses["rag"] = f"Error: {str(e)}"

        try:
            responses["finetuned_rag"] = self.chat_finetuned_rag(question)
        except Exception as e:
            logger.error(f"Error getting fine-tuned RAG response: {e}")
            responses["finetuned_rag"] = f"Error: {str(e)}"

        # Update conversation history (use RAG response as the canonical one)
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=responses["rag"]))

        # Trim history if too long
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2 :]

        return responses

    def chat(self, question: str, mode: str = "rag") -> str:
        """
        Process a user message and return a response.

        Args:
            question: The user's input message
            mode: Response mode ("base", "rag", or "finetuned_rag")

        Returns:
            The chatbot's response
        """
        if mode == "base":
            response = self.chat_base(question)
        elif mode == "finetuned_rag":
            response = self.chat_finetuned_rag(question)
        else:
            response = self.chat_rag(question)

        # Update conversation history
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=response))

        # Trim history
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2 :]

        return response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> list[dict[str, str]]:
        """Get the conversation history as a list of dictionaries."""
        history = []
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
