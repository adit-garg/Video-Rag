import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="YouTube Video Q&A Assistant",
    page_icon="🎥",
    layout="wide"
)

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]  
        
        elif 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        
        return None
    except:
        return None

def get_transcript_with_fallback(video_id):
    translator = GoogleTranslator()
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, "en"
    
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            for transcript_info in transcript_list:
                try:
                    transcript_data = transcript_info.fetch()
                    original_text = " ".join(chunk["text"] for chunk in transcript_data)
                    
                    if transcript_info.language_code != 'en':
                        st.info(f"Translating from {transcript_info.language} to English...")
                        chunks = [original_text[i:i+4000] for i in range(0, len(original_text), 4000)]
                        translated_chunks = []
                        
                        for chunk in chunks:
                            try:
                                translated = translator.translate(chunk, dest='en', src=transcript_info.language_code)
                                translated_chunks.append(translated.text)
                            except:
                                translated_chunks.append(chunk)
                        
                        transcript = " ".join(translated_chunks)
                        return transcript, transcript_info.language_code
                    else:
                        return original_text, transcript_info.language_code
                        
                except Exception as e:
                    continue
            
            raise Exception("No usable transcript found")
            
        except Exception as e:
            raise Exception(f"Could not retrieve transcript: {str(e)}")

def create_enhanced_retrieval_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": 6, 
            "fetch_k": 12,  
            "lambda_mult": 0.7  
        }
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    prompt = PromptTemplate(
        template="""You are an expert assistant analyzing YouTube video content. Your task is to provide accurate, comprehensive answers based ONLY on the video transcript provided.

CONTEXT FROM VIDEO TRANSCRIPT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using information from the provided transcript context
2. If the context doesn't contain enough information to answer the question, clearly state "I don't have enough information from the video transcript to answer this question"
3. Structure your response clearly with main points and supporting details
4. Use specific examples or quotes from the video when relevant
5. If the question asks about something not covered in the video, be honest about the limitations
6. Keep your answer concise but comprehensive
7. If you find contradictory information in the transcript, acknowledge it

ANSWER:""",
        input_variables=['context', 'question']
    )
    
    def format_docs(retrieved_docs):
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[Context {i+1}]: {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    
    main_chain = parallel_chain | prompt | llm | parser
    
    return main_chain

def process_video(video_url):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            return False
        
        with st.spinner("Fetching video transcript..."):
            transcript, source_lang = get_transcript_with_fallback(video_id)
            
            if source_lang != 'en':
                st.success(f"Successfully retrieved and translated transcript from {source_lang} to English")
            else:
                st.success("Successfully retrieved English transcript")
        
        with st.spinner("Processing transcript and creating knowledge base..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, 
                chunk_overlap=150, 
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            chunks = splitter.create_documents([transcript])
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            st.session_state.vector_store = vector_store
            st.session_state.video_title = f"Video ID: {video_id}"
            
        st.success("Video processed successfully! You can now ask questions about the content.")
        return True
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return False

def main():
    st.title("🎥 YouTube Video Q&A Assistant")
    st.markdown("Upload a YouTube video URL and ask questions about its content!")
    
    with st.sidebar:
        st.header("📹 Video Input")
        video_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        if st.button("Process Video", type="primary"):
            if video_url:
                if process_video(video_url):
                    st.session_state.chat_history = [] 
            else:
                st.error("Please enter a YouTube URL")
        
        if st.session_state.vector_store is not None:
            st.success("Video processed and ready for questions!")
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    if st.session_state.vector_store is not None:
        st.header("Ask Questions About the Video")
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
        
        question = st.chat_input("Ask a question about the video content...")
        
        if question:
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chain = create_enhanced_retrieval_chain(st.session_state.vector_store)
                        answer = chain.invoke(question)
                        st.write(answer)
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer
                        })
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
    
    else:
        st.info("Please enter a YouTube URL in the sidebar to get started!")

if __name__ == "__main__":
    main()
