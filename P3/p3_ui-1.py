import streamlit as st
import asyncio
# from p3_mcp_client import OllamaMCPClient
from rag_client_for_p3 import OllamaMCPClient
import pyttsx3
import speech_recognition as sr
import threading
from pathlib import Path
import os

# TTS (Text-to-Speech) function
def speak_text(text_to_speak: str):
    def speak():
        """The actual work to be done in the thread."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text_to_speak)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS Error] Could not speak: {e}")

    tts_thread = threading.Thread(target=speak, daemon=True)
    tts_thread.start()

# STT (Speech-to-Text) function 
def listen_to_microphone():
    """Listen to microphone and return recognized text."""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.session_state.listening = True
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            st.session_state.listening = False
            
        # Try to recognize speech using Google's speech recognition
        try:
            text = recognizer.recognize_google(audio)
            return text, None
        except sr.UnknownValueError:
            return None, "Could not understand audio"
        except sr.RequestError as e:
            # Fallback to Sphinx if Google fails
            try:
                text = recognizer.recognize_sphinx(audio)
                return text, None
            except:
                return None, f"Speech recognition error: {e}"
                
    except sr.WaitTimeoutError:
        st.session_state.listening = False
        return None, "Listening timeout - no speech detected"
    except Exception as e:
        st.session_state.listening = False
        return None, f"Microphone error: {e}"

# Environment file management
def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path(".env")
    env_vars = {}
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    return env_vars

def save_env_file(env_vars):
    """Save environment variables to .env file"""
    env_path = Path(".env")
    
    with open(env_path, 'w') as f:
        f.write("# API Keys\n")
        if 'OPENROUTER_API_KEY' in env_vars:
            f.write(f"OPENROUTER_API_KEY={env_vars['OPENROUTER_API_KEY']}\n")
        if 'OLLAMA_API_KEY' in env_vars:
            f.write(f"OLLAMA_API_KEY={env_vars['OLLAMA_API_KEY']}\n")
        
        f.write("# Directories\n")
        if 'DATA_DIR' in env_vars:
            f.write(f"DATA_DIR={env_vars['DATA_DIR']}\n")
        if 'OUTPUT_DIR' in env_vars:
            f.write(f"OUTPUT_DIR={env_vars['OUTPUT_DIR']}\n")
        if 'RAG_STORAGE_DIR' in env_vars:
            f.write(f"RAG_STORAGE_DIR={env_vars['RAG_STORAGE_DIR']}\n")
    
    return True

# Connect to LLM MCP Client
async def connect_to_mcp_client(mcp_server_url: str, llm_model_name: str, embedding_model_name: str, vlm_model_name: str):
    """Create the MCP client, connect to the MCP server, and initialize server tools."""
    
    env_vars   = load_env_file()
    if not env_vars:
        print("Error in reading the .env file")
        return None, False, "Error in reading the .env file"

    client = OllamaMCPClient(
        llm_model=llm_model_name,
        embedding_model= embedding_model_name,
        vlm_model= vlm_model_name,
        server_url = mcp_server_url,        
        env_vars   = load_env_file()
    )

    success, message = await client.initialize_tools()
    return client, success, message

# Streamlit UI
def init_session_state():
    """Initialize session state variables."""
    if "client" not in st.session_state:
        st.session_state.client = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "server_url" not in st.session_state:
        st.session_state.server_url = "http://127.0.0.1:3000/mcp"
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = "gpt-oss:20b-cloud"
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = "nomic-embed-text"
    if "vlm_model_name" not in st.session_state:
        st.session_state.vlm_model_name = "openrouter/polaris-alpha"
    if "audio_enabled" not in st.session_state:
        st.session_state.audio_enabled = False
    if "speech_input" not in st.session_state:
        st.session_state.speech_input = ""
    if "listening" not in st.session_state:
        st.session_state.listening = False
    
    # Load environment variables
    env_vars = load_env_file()
    
    # API Keys - MUST be in .env file
    if "openrouter_api_key" not in st.session_state:
        openrouter_key = env_vars.get('OPENROUTER_API_KEY', '')
        if not openrouter_key:
            st.session_state.config_error = "OPENROUTER_API_KEY not found in .env file"
        st.session_state.openrouter_api_key = openrouter_key
    
    if "ollama_api_key" not in st.session_state:
        ollama_key = env_vars.get('OLLAMA_API_KEY', '')
        if not ollama_key:
            st.session_state.config_error = "OLLAMA_API_KEY not found in .env file"
        st.session_state.ollama_api_key = ollama_key
    
    # Directory settings - MUST be in .env file
    if "data_dir" not in st.session_state:
        data_dir = env_vars.get('DATA_DIR', '')
        if not data_dir:
            st.session_state.config_error = "DATA_DIR not found in .env file"
        st.session_state.data_dir = data_dir
    
    if "output_dir" not in st.session_state:
        output_dir = env_vars.get('OUTPUT_DIR', '')
        if not output_dir:
            st.session_state.config_error = "OUTPUT_DIR not found in .env file"
        st.session_state.output_dir = output_dir
    
    if "rag_storage_dir" not in st.session_state:
        rag_storage_dir = env_vars.get('RAG_STORAGE_DIR', '')
        if not rag_storage_dir:
            st.session_state.config_error = "RAG_STORAGE_DIR not found in .env file"
        st.session_state.rag_storage_dir = rag_storage_dir
    
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    if "config_error" not in st.session_state:
        st.session_state.config_error = None

# Main Method
def main():
    st.set_page_config(page_title="Knowledge Navigator", page_icon="🤖", layout="wide")

    init_session_state()
    
    # Check for configuration errors
    if st.session_state.config_error:
        st.error(f"⚠️ Configuration Error: {st.session_state.config_error}")
        st.error("Please create a .env file with all required settings.")
        
        st.markdown("### Required .env file format:")
        st.code("""# API Keys
            OPENROUTER_API_KEY=your_openrouter_key_here
            OLLAMA_API_KEY=your_ollama_key_here

            # Directories
            DATA_DIR=/path/to/your/my_files
            OUTPUT_DIR=/path/to/your/output
            RAG_STORAGE_DIR=/path/to/your/rag_storage
            """, 
        language="bash")
        
        st.markdown("### Example .env file:")
        st.code("""# API Keys
            OPENROUTER_API_KEY=sk-or-v1-abc123...
            OLLAMA_API_KEY=c7f1e8e3cc184caa...

            # Directories
            DATA_DIR=/Users/yourname/Documents/project/my_files
            OUTPUT_DIR=/Users/yourname/Documents/project/output
            RAG_STORAGE_DIR=/Users/yourname/Documents/project/rag_storage
            """, 
        language="bash")
        
        st.info("💡 After creating the .env file in the same directory as this script, refresh this page.")
        st.stop()  # Stop execution until .env is properly configured

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")        

        # Audio Toggle Section
        st.subheader("🔊 Audio Settings")
        audio_toggle = st.toggle(
            "Enable Text-to-Speech",
            value=st.session_state.audio_enabled,
            help="Toggle text-to-speech for assistant responses"
        )
        st.session_state.audio_enabled = audio_toggle
        
        if audio_toggle:
            st.info("🔊 Audio is enabled")            
        else:
            st.info("🔇 Audio is disabled")
        
        st.divider()

        # Connection Settings
        st.subheader("🔌 Connection")
        
        server_url = st.text_input(
            "MCP Server URL",
            value=st.session_state.server_url,
            help="URL of your MCP server"
        )

        llm_model_name = st.text_input(
            "Ollama Model",
            value=st.session_state.llm_model_name,
            help="Name of the llm model to use"
        )

        embedding_model_name = st.text_input(
            "Embedding Model",
            value=st.session_state.embedding_model_name,
            help="Name of text embedding model to use"
        )

        vlm_model_name = st.text_input(
            "VLM Model",
            value=st.session_state.vlm_model_name,
            help="Name of the vision-language model to use"
        )

        if st.button("Connect", type="primary", use_container_width=True):
            with st.spinner("Connecting to MCP server..."):
                client, success, message = asyncio.run(
                    connect_to_mcp_client(
                        server_url, 
                        llm_model_name,
                        embedding_model_name,
                        vlm_model_name
                    )
                )
                if success:
                    st.session_state.client = client
                    st.session_state.connected = True
                    st.session_state.server_url = server_url
                    
                    # update the model names in the session variables
                    st.session_state.llm_model_name = llm_model_name
                    st.session_state.embedding_model_name = embedding_model_name
                    st.session_state.vlm_model_name = vlm_model_name
                    
                    st.success(message)
                else:
                    st.error(message)

        if st.session_state.connected:
            st.success("✅ Connected")
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.client:
                    st.session_state.client.messages = [
                        {"role": "system", "content": st.session_state.client.system_prompt}
                    ]
                st.rerun()
        else:
            st.warning("⚠️ Not connected")

        st.divider()
        
        # API Keys and Directory Configuration
        st.subheader("🔑 Configuration")
        
        # Expandable section for API keys
        with st.expander("API Keys", expanded=False):
            st.markdown("**OpenRouter API Key** (for VLM)")
            openrouter_key = st.text_input(
                "OpenRouter Key",
                value=st.session_state.openrouter_api_key,
                type="password",
                help="API key for OpenRouter (VLM)",
                key="openrouter_key_input",
                label_visibility="collapsed"
            )
            
            st.markdown("**Ollama Cloud API Key**")
            ollama_key = st.text_input(
                "Ollama Key",
                value=st.session_state.ollama_api_key,
                type="password",
                help="API key for Ollama Cloud",
                key="ollama_key_input",
                label_visibility="collapsed"
            )
            
            if st.button("💾 Save API Keys", use_container_width=True):
                if not openrouter_key or not ollama_key:
                    st.error("Both API keys are required!")
                else:
                    st.session_state.openrouter_api_key = openrouter_key
                    st.session_state.ollama_api_key = ollama_key
                    
                    # Save to environment
                    env_vars = load_env_file()
                    env_vars['OPENROUTER_API_KEY'] = openrouter_key
                    env_vars['OLLAMA_API_KEY'] = ollama_key
                    
                    if save_env_file(env_vars):
                        # Also set in current environment
                        os.environ['OPENROUTER_API_KEY'] = openrouter_key
                        os.environ['OLLAMA_API_KEY'] = ollama_key
                        st.success("✅ API keys saved to .env file")
                        st.session_state.config_error = None
                    else:
                        st.error("❌ Failed to save API keys")
        
        # Expandable section for directories
        with st.expander("Directory Settings", expanded=False):
            st.markdown("**Data Directory** (PDF files)")
            data_dir = st.text_input(
                "Path to data folder",
                value=st.session_state.data_dir,
                help="Directory containing your PDF files",
                key="data_dir_input"
            )
            
            st.markdown("**Output Directory** (Parsed documents)")
            output_dir = st.text_input(
                "Path to output folder",
                value=st.session_state.output_dir,
                help="Directory for parsed document outputs",
                key="output_dir_input"
            )
            
            st.markdown("**RAG Storage** (Vector database)")
            rag_storage_dir = st.text_input(
                "Path to RAG storage folder",
                value=st.session_state.rag_storage_dir,
                help="Directory for persistent RAG storage",
                key="rag_storage_input"
            )
            
            if st.button("💾 Save Directories", use_container_width=True):
                if not data_dir or not output_dir or not rag_storage_dir:
                    st.error("All directory paths are required!")
                else:
                    st.session_state.data_dir = data_dir
                    st.session_state.output_dir = output_dir
                    st.session_state.rag_storage_dir = rag_storage_dir
                    
                    # Create directories if they don't exist
                    try:
                        Path(data_dir).mkdir(parents=True, exist_ok=True)
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        Path(rag_storage_dir).mkdir(parents=True, exist_ok=True)
                        
                        # Save to environment
                        env_vars = load_env_file()
                        env_vars['DATA_DIR'] = data_dir
                        env_vars['OUTPUT_DIR'] = output_dir
                        env_vars['RAG_STORAGE_DIR'] = rag_storage_dir
                        
                        if save_env_file(env_vars):
                            st.success("✅ Directories saved and created")
                            st.session_state.config_error = None
                        else:
                            st.error("❌ Failed to save directories")
                    except Exception as e:
                        st.error(f"❌ Failed to create directories: {e}")
            
            # Display current directories
            st.markdown("---")
            st.markdown("**Current Settings:**")
            st.code(f"""
                Data: {st.session_state.data_dir}
                Output: {st.session_state.output_dir}
                Storage: {st.session_state.rag_storage_dir}
            """)

        st.divider()
        st.markdown("""
            ### 💡 Tips
            - Configure .env file first
            - Connect to your MCP server
            - Toggle audio for voice feedback
            - Use 🎤 button for voice input
            - Ask questions or request tool usage
            - Responses stream in real-time
            - Tool calls are executed automatically
        """)

    # Main chat interface
    st.markdown("### I'm P3, Your Knowledge Navigator")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Microphone button above chat input
    mic_col1, mic_col2, mic_col3 = st.columns([1, 6, 1])
    
    with mic_col2:
        if st.button(
            "🎤 Click to Speak", 
            help="Click to use voice input", 
            disabled=not st.session_state.connected, 
            use_container_width=True
        ):
            if st.session_state.listening:
                st.warning("Already listening...")
            else:
                with st.spinner("🎤 Listening... Speak now!"):
                    text, error = listen_to_microphone()
                    
                    if text:
                        st.session_state.speech_input = text
                        st.success(f"✅ Recognized: {text}")
                        st.rerun()
                    elif error:
                        st.error(error)

    # chat_input ALWAYS rendered
    prompt = st.chat_input(
        "Type your message here… or use 🎤 above",
        disabled=not st.session_state.connected,
        key="chat_input"
    )

    # If we have speech input, override the prompt
    if st.session_state.speech_input:
        prompt = st.session_state.speech_input
        st.session_state.speech_input = ""

    if prompt:
        if not st.session_state.connected:
            st.error("Please connect to the MCP server first!")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            try:
                # Use the synchronous generator (no async context managers)
                full_response = st.write_stream(
                    st.session_state.client.chat_stream(prompt)
                )
                
                # Only speak if audio is enabled
                if st.session_state.audio_enabled:
                    try:
                        speak_text(full_response)
                    except Exception as tts_error:
                        st.warning(f"TTS Error: {tts_error}")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()