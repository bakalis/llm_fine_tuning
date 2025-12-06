import gradio as gr
from llama_cpp import Llama
import whisper
from TTS.api import TTS
import tempfile
import os
from huggingface_hub import hf_hub_download

# Download GGUF files to cache directory
finetuned_1b_llama_model_path = hf_hub_download(
    repo_id="bakalis/Fine_Tuned_Llama_3.2_1B_Q4_K_M",
    filename="Fine_Tuned_Llama_3.2_1B_Q4_K_M.gguf",
    token=os.environ["HF_TOKEN"],
    cache_dir="models"
)

finetuned_3b_llama_model_path = hf_hub_download(
    repo_id="bakalis/Fine_Tuned_Llama_3.2_3B_Q4_K_M",
    filename="Fine_Tuned_Llama_3.2_3B_Q4_K_M.gguf",
    token=os.environ["HF_TOKEN"],
    cache_dir="models"
)

llms = {
    'Fine-tuned 1B Llama 3.2': Llama(
        model_path=finetuned_1b_llama_model_path,
        n_ctx=8192,
        n_threads=2,
        n_gpu_layers=0
    ),
    'Fine-tuned 3B Llama 3.2': Llama(
        model_path=finetuned_3b_llama_model_path,
        n_ctx=8192,
        n_threads=2,
        n_gpu_layers=0
    )
}

whisper_model = whisper.load_model("small.en")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def speak_last_response(history):
    if not history:
        return None
    
    last_response = history[-1]['content']
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        tts.tts_to_file(text=last_response, file_path=temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def audio_to_text(audio_path):
    if not audio_path:
        return ""
    try:
        result = whisper_model.transcribe(audio_path)
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Cleaned up audio file: {audio_path}")
            except Exception as e:
                print(f"Error cleaning up audio file: {e}")
        
        return result["text"]
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def make_chatbot_messages(history):
    return [{"role": msg["role"], "content": msg["content"]} for msg in history if msg["role"] in ["user", "assistant"]]

def deduplicate_history(history):
    """
    Deduplicate incremental assistant messages (placeholders or streaming updates)
    and keep only the final full assistant message.
    """
    clean_history = []
    for i, msg in enumerate(history):
        if msg["role"] == "assistant":
            # Only keep the last assistant message in a consecutive sequence
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                continue
        clean_history.append(msg)
    return clean_history

def text_submit(message, history, selected_model):
    if not message:
        yield make_chatbot_messages(history), history, None, selected_model
        return
    
    llm = llms[selected_model]

    print(f"Inference using selected model: {selected_model}")

    # add a ... placeholder in the llm's response to indicate that an answer is being generated
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "..."})
    chatbot_messages = make_chatbot_messages(history)
    yield chatbot_messages, history, None, selected_model

    response = ""
    for chunk in llm.create_chat_completion(
        messages=history[:-1],  # exclude placeholder when generating
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stream=True
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            response += delta["content"]
            history[-1]["content"] = response
            chatbot_messages = make_chatbot_messages(history)
            yield chatbot_messages, history, None, selected_model # streaming llm's response token by token

    # Yield full deduplicated history after completion
    history[-1]["content"] = response
    clean_history = deduplicate_history(history)
    yield make_chatbot_messages(clean_history), clean_history, None, selected_model

def audio_submit(audio_path, history, selected_model):
    if not audio_path:
        yield make_chatbot_messages(history), history, None, selected_model
        return
    
    llm = llms[selected_model]

    print(f"Inference using selected model: {selected_model}")
    text = audio_to_text(audio_path)
    print(f"Transcribed text: {text}")
    if not text:
        yield make_chatbot_messages(history), history, None, selected_model
        return

    # add a ... placeholder in the llm's response to indicate that an answer is being generated
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": "..."})
    chatbot_messages = make_chatbot_messages(history)
    yield chatbot_messages, history, None, selected_model

    response = ""
    for chunk in llm.create_chat_completion(
        messages=history[:-1],  # exclude placeholder when generating
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stream=True
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            response += delta["content"]
            history[-1]["content"] = response
            chatbot_messages = make_chatbot_messages(history)
            yield chatbot_messages, history, None, selected_model  # streaming llm's response token by token

    # Step 3: Yield full deduplicated history after completion
    history[-1]["content"] = response
    clean_history = deduplicate_history(history)
    yield make_chatbot_messages(clean_history), clean_history, None, selected_model

with gr.Blocks() as demo:
    gr.Markdown("## Fine-tuned Llama 3.2 Chatbot with Audio & Text Input")

    default_model = 'Fine-tuned 1B Llama 3.2'

    chat_history = gr.State([])  # stores conversation history
    model_state = gr.State(default_model)  # default selected model

    chatbot = gr.Chatbot(height='50vh')

    model_selector = gr.Dropdown(
        label="Select Model",
        choices=list(llms.keys()),
        value=default_model
    )

    chat_textbox = gr.Textbox(
        placeholder="Type a message...",
        label="Talk to Llama"
    )

    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record audio")
    
    with gr.Row():
        tts_button = gr.Button("🔊 Play Last Response (R)", variant="secondary")
    
    audio_output = gr.Audio(label="TTS Output", autoplay=True, visible=True)

    chat_textbox.submit(
        text_submit, 
        inputs=[chat_textbox, chat_history, model_state], 
        outputs=[chatbot, chat_history, chat_textbox, model_state]
    )
    
    audio_input.change(
        audio_submit, 
        inputs=[audio_input, chat_history, model_state], 
        outputs=[chatbot, chat_history, audio_input, model_state]
    )
    
    model_selector.change(lambda m: m, model_selector, model_state)

    tts_button.click(
        speak_last_response, 
        inputs=[chat_history],
        outputs=[audio_output]
    )

    # Add keyboard shortcut using JavaScript
    demo.load(
        None,
        None,
        None,
        js="""
            function() {
                document.addEventListener('keydown', function(event) {
                    // Check if the user is typing in an input field or textarea
                    const activeElement = document.activeElement;
                    const isTyping = activeElement.tagName === 'INPUT' || 
                                activeElement.tagName === 'TEXTAREA' ||
                                activeElement.isContentEditable;
                    
                    // Only trigger TTS if not typing and 'R' is pressed
                    if ((event.key === 'r' || event.key === 'R') && !isTyping) {
                        event.preventDefault(); // Prevent default 'r' behavior
                        // Find and click the TTS button
                        const buttons = document.querySelectorAll('button');
                        for (let btn of buttons) {
                            if (btn.textContent.includes('Play Last Response')) {
                                btn.click();
                                break;
                            }
                        }
                    }
                });
            }
        """
    )

if __name__ == "__main__":
    demo.launch()