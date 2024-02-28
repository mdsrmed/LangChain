# Api integration

import os 
from dotenv import load_dotenv
from langchain.llms import OpenAI 
import streamlit as st 
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory


load_dotenv()

# Streamlit framework 
st.title("Langchain Openai SearchBot")
input_text=st.text_input("Search the topic")

#Prompt Templates

first_input_prompt = PromptTemplate( 
        input_variables = ['topic'],
        template = "Tell me about {topic}"                                    
)

#Memory

topic_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')
details_memory = ConversationBufferMemory(input_key = 'details', memory_key = 'chat_history')
#more_details_memory = ConversationBufferMemory(input_key = 'more_details', memory_key = 'chat_history')

# OpenAI LLMS
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm = llm, prompt = first_input_prompt, verbose=True, output_key="details", memory = topic_memory)

#Prompt Templates

second_input_prompt = PromptTemplate( 
        input_variables = ['details'],
        template = "Tell me more about {details}"                                    
)

chain2 = LLMChain(llm = llm, prompt = first_input_prompt, verbose=True, output_key="more details", memory = details_memory)

final_chain = SequentialChain(chains = [chain,chain2],input_variables = ['topic'],output_variables = ['details','more details'], verbose=True)

if input_text:
    st.write(final_chain({'topic':input_text}))
    
    with st.expander("Details"):
        st.info(details_memory.buffer)

