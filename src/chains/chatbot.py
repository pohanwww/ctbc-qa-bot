"""
CTBC Chatbot Chain.

Implements the main chatbot logic using LangChain, combining:
- RAG retrieval for CTBC FAQ knowledge
- Conversation history management
- Prompt engineering for customer service behavior
- Safety guardrails and response filtering
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.llm.loader import LLMWrapper
from src.rag.retriever import FAQRetriever

logger = logging.getLogger(__name__)


# System prompt for the CTBC customer service chatbot
SYSTEM_PROMPT = """You are a helpful, professional, and friendly English-speaking customer service agent for CTBC Bank (中國信託商業銀行).

Your role is to assist customers with questions about CTBC Bank's products, services, fees, and policies.

IMPORTANT GUIDELINES:

1. **Use FAQ Context**: Base your factual answers ONLY on the provided FAQ context below. Do not make up information about CTBC Bank's specific products, rates, fees, or policies.

2. **Language**: Always respond in clear, professional English. If the FAQ context is in Chinese, translate and explain it clearly in English.

3. **When Information is Missing**: If the FAQ context does not contain relevant information to answer the question:
   - Acknowledge that you don't have that specific information
   - Suggest the customer contact CTBC Bank directly:
     - Customer Service Hotline: 0800-024-365 (Taiwan)
     - International: +886-2-2745-8080
     - Visit: www.ctbcbank.com

4. **Safety and Compliance**:
   - Do NOT provide specific financial, legal, or tax advice
   - Do NOT make promises about loan approvals, interest rates, or account decisions
   - Always recommend consulting with a CTBC Bank representative for complex matters
   - Include appropriate disclaimers when discussing financial products

5. **Tone**: Be warm, patient, and helpful. Show empathy for customer concerns.

6. **Format**: Keep responses concise but complete. Use bullet points for lists when appropriate.

---

FAQ CONTEXT:
{context}

---

Remember: Your knowledge about CTBC Bank comes ONLY from the FAQ context above. For anything not covered, guide customers to official CTBC Bank channels."""


class CTBCChatbot:
    """
    Main chatbot class for CTBC customer service.

    This class orchestrates:
    - RAG-based knowledge retrieval
    - Conversation history management
    - Prompt construction and LLM invocation
    - Response post-processing

    Attributes:
        llm: The language model wrapper
        retriever: The FAQ retriever
        conversation_history: List of conversation messages
        max_history_turns: Maximum number of conversation turns to keep
    """

    def __init__(
        self,
        llm: LLMWrapper,
        retriever: FAQRetriever,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        max_history_turns: int = 10,
    ):
        """
        Initialize the CTBC chatbot.

        Args:
            llm: The language model wrapper
            retriever: The FAQ retriever instance
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            max_history_turns: Maximum conversation turns to maintain
        """
        self.llm = llm
        self.retriever = retriever
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_history_turns = max_history_turns

        # Conversation history as list of messages
        self.conversation_history: list[HumanMessage | AIMessage] = []

        logger.info("CTBC Chatbot initialized")

    def _build_prompt(self, user_message: str, context: str) -> str:
        """
        Build the full prompt for the LLM.

        Args:
            user_message: The user's current message
            context: Retrieved FAQ context

        Returns:
            The complete prompt string
        """
        # Build system message with context
        system_content = SYSTEM_PROMPT.format(context=context)

        # Build conversation history string
        history_str = ""
        if self.conversation_history:
            history_parts = []
            for msg in self.conversation_history[-self.max_history_turns * 2 :]:
                if isinstance(msg, HumanMessage):
                    history_parts.append(f"Customer: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_parts.append(f"Agent: {msg.content}")
            history_str = "\n".join(history_parts)

        # Construct the prompt using the model's chat template format
        # For Qwen3, we use the standard chat format (same as Qwen2.5)
        prompt_parts = [
            f"<|im_start|>system\n{system_content}<|im_end|>",
        ]

        # Add conversation history
        for msg in self.conversation_history[-self.max_history_turns * 2 :]:
            if isinstance(msg, HumanMessage):
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

        # Add current user message
        prompt_parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def _postprocess_response(self, response: str) -> str:
        """
        Post-process the LLM response.

        Args:
            response: Raw LLM output

        Returns:
            Cleaned response string
        """
        # Remove any trailing special tokens
        response = response.strip()

        # Remove incomplete sentences at the end (if response was cut off)
        if response and response[-1] not in ".!?。！？\"'":
            # Try to find the last complete sentence
            for punct in [". ", "! ", "? ", "。", "！", "？"]:
                last_punct = response.rfind(punct)
                if last_punct > len(response) * 0.5:  # Only trim if we keep most of the response
                    response = response[: last_punct + 1]
                    break

        # Remove any remaining special tokens
        special_tokens = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        for token in special_tokens:
            response = response.replace(token, "")

        return response.strip()

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main method for chatbot interaction:
        1. Retrieves relevant FAQ context
        2. Builds the prompt with history and context
        3. Generates a response using the LLM
        4. Updates conversation history

        Args:
            user_message: The user's input message

        Returns:
            The chatbot's response
        """
        logger.debug(f"Processing user message: {user_message[:50]}...")

        # Retrieve relevant FAQ context
        context = self.retriever.get_relevant_context(user_message)
        logger.debug(f"Retrieved context: {context[:100]}...")

        # Build prompt
        prompt = self._build_prompt(user_message, context)

        # Generate response
        try:
            raw_response = self.llm.invoke(prompt)
            response = self._postprocess_response(raw_response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = (
                "I apologize, but I'm experiencing technical difficulties. "
                "Please try again or contact CTBC Bank directly at 0800-024-365."
            )

        # Update conversation history
        self.conversation_history.append(HumanMessage(content=user_message))
        self.conversation_history.append(AIMessage(content=response))

        # Trim history if too long
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2 :]

        return response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> list[dict[str, str]]:
        """
        Get the conversation history as a list of dictionaries.

        Returns:
            List of {"role": "user"|"assistant", "content": str} dicts
        """
        history = []
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def set_context_override(self, context: str) -> None:
        """
        Set a context override for testing or special scenarios.

        Args:
            context: Custom context to use instead of RAG retrieval
        """
        # TODO: Implement context override functionality
        pass
