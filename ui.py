import streamlit as st
from cohere_rag import Chatbot, Documents
import cohere


st.title("Cohere RAG Chatbot")
st.markdown(
    "This is a chatbot that uses Cohere's RAG model to answer questions about building a growth mindset.")

with st.sidebar:
    st.title("Growth Mindset Chatbot")
    st.markdown("This chatbot can provide information on how to enhance performance by applying a growth mindset, how to learn skills faster, and how to embrace the benefits of stress. It uses the embedding, rerank, and chat endpoints of the Cohere API to retrieve documents and generate responses.")
    cohere_api_key = st.text_input("Cohere API Key", type="password")
    st.divider()
    st.markdown("Made by Ethan Benjamin")
# create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


sources = [
    {
        "title": "How to Enhance Performance by Applying a Growth Mindset",
        "url": "https://www.wisdominanutshell.academy/andrew-huberman/how-to-enhance-performance-learning-by-applying-a-growth-mindset-huberman-lab-podcast/"
    },
    # {
    #     "title": "A synergistic mindsets intervention protects adolescents from stress",
    #     "url": "https://www.nature.com/articles/s41586-022-04907-7"}, # this article is so long
    {
        "title": "Mindset Matters: How to Embrace the Benefits of Stress",
        "url": "https://www.gsb.stanford.edu/insights/mindset-matters-how-embrace-benefits-stress"},
    {
        "title": "Growth Mindset Summary",
        "url": "https://rickkettner.com/mindset-book-summary/"}
]


def is_cohere_api_key_valid(api_key: str) -> bool:
    """
    Checks if the given API key is valid.
    """
    if len(api_key) == 0:
        st.info(
            "Please enter your Cohere API key in the sidebar to start using the chatbot.", icon="🔥")
        return False

    try:
        co = cohere.Client(api_key)
        co.generate("Hello!")
        return True
    except cohere.CohereAPIError:
        st.warning(
            "An error occurred while initializing the Cohere client. Please check your API key.", icon="🚨")
        return False


if is_cohere_api_key_valid(cohere_api_key):
    # Initialize the Cohere API client
    client = cohere.Client(cohere_api_key)
    # Create an instance of the Documents class with the given sources
    documents = Documents(sources, client)
    # Create an instance of the Chatbot class with the Documents instance
    chatbot = Chatbot(documents, client)
    # Chat Interface
    if prompt := st.chat_input("What's up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                response = chatbot.generate_response(prompt)
                text = st.write_stream(response)
            except cohere.CohereAPIError as e:
                text = "Sorry, I'm limited to 10 API calls per minute. Please try again later."
                st.error(e)
        # Add assistant message to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": text})
