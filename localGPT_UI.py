import streamlit as st
from run_localGPT import setup_qa
from streamlit_extras.add_vertical_space import add_vertical_space

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Converse with your Data')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LocalGPT](https://github.com/PromtEngineer/localGPT) 
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')

if "QA" not in st.session_state:
    st.session_state["QA"] = setup_qa()

st.title('LocalGPT App 💬')
    # Create a text input box for the user
prompt = st.text_input('Input your prompt here')
# while True:

    # If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = st.session_state["QA"](prompt)
    answer, docs = response["result"], response["source_documents"]
    # ...and write it out to the screen
    st.write(answer)

    # With a streamlit expander  
    with st.expander('Source documents'):
        # Write out the first
        for doc in docs: 
            st.write(f"Source Document: {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content) 
            st.write("--------------------------------")