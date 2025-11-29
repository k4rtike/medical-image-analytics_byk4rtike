import streamlit as st
from pathlib import Path
import google.generativeai as genai

# Configure genAI with API key
api_key = None
try:
    from api_key import api_key
except ImportError:
    pass

# Try to get from st.secrets if not found in api_key.py or if we want to prioritize secrets
# Note: accessing st.secrets might fail if no secrets.toml exists locally, 
# but usually it returns an empty dict-like object or raises FileNotFoundError depending on version.
# Safest way:
try:
    if not api_key and "api_key" in st.secrets:
        api_key = st.secrets["api_key"]
except (FileNotFoundError, KeyError):
    pass

if not api_key:
    st.error("API Key not found. Please set it in Streamlit secrets or api_key.py")
    st.stop()

genai.configure(api_key=api_key)

# Set up the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

system_prompt = """
You are a highly skilled medical imaging expert and analyst. Your task is to examine the uploaded medical image and provide a detailed analysis.

Please structure your response as follows:
1.  **Detailed Description**: Describe what you see in the image (modality, body part, visible structures).
2.  **Analysis**: Identify any potential abnormalities, diseases, or conditions visible in the image.
3.  **Potential Dangers**: Highlight any critical or urgent findings that require immediate attention.
4.  **Disclaimer**: Always include a disclaimer that you are an AI assistant and this is not a substitute for professional medical advice.
"""

model = genai.GenerativeModel(
    model_name="gemini-flash-latest",
    generation_config=generation_config,
    system_instruction=system_prompt,
)

st.set_page_config(page_title="DiseaseImage Analytics", page_icon=":robot:")

# Set the title
st.title("Medical Image Analytics")
st.subheader("An application that can help users to identify medical images")

uploaded_file = st.file_uploader("Upload the medical image for analysis", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, width=300, caption="Uploaded Image")
    
    # Store the uploaded file in session state to persist across reruns
    if "uploaded_file_content" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_content = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_type = uploaded_file.type
        # Reset history when a new file is uploaded
        st.session_state.history = []

submit_button = st.button("Generate the analysis")

if "history" not in st.session_state:
    st.session_state.history = []

if submit_button and uploaded_file:
    try:
        image_data = st.session_state.uploaded_file_content
        mime_type = st.session_state.uploaded_file_type
        
        image_parts = [
            {
                "mime_type": mime_type,
                "data": image_data
            }
        ]
        
        prompt_parts = [
            image_parts[0],
            "\nAnalyze this medical image based on the system instructions."
        ]
        
        with st.spinner("Analyzing the image..."):
            response = model.generate_content(prompt_parts)
            st.subheader("Analysis Result")
            st.write(response.text)
            
            # Add to history
            st.session_state.history.append({"role": "user", "parts": ["Analyze this medical image."]})
            st.session_state.history.append({"role": "model", "parts": [response.text]})
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
elif submit_button and not uploaded_file:
    st.warning("Please upload an image first.")

# Display chat history if there is any
if st.session_state.history:
    st.divider()
    st.subheader("Follow-up Questions")
    
    # Display previous chat messages (excluding the initial analysis request/response if desired, but keeping for context)
    # We'll show the conversation from the user's perspective
    for message in st.session_state.history:
        if message["role"] == "user" and message["parts"][0] == "Analyze this medical image.":
            continue # Skip the initial trigger message in the chat view
        if message["role"] == "model" and message["parts"][0].startswith("1.  **Detailed Description**"):
             continue # Skip the initial analysis in the chat view as it's already shown above (or we can show it)
             # Actually, let's just show the follow-up conversation to avoid clutter, 
             # or show everything. Let's show everything for clarity but maybe skip the big analysis block if it's redundant.
             # For now, let's just show the follow-up Q&A.
        
        with st.chat_message(message["role"]):
            st.markdown(message["parts"][0])

    # Chat input for follow-up questions
    if prompt := st.chat_input("Ask a follow-up question about the image..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare the chat session with history
        # We need to reconstruct the chat history for Gemini, including the image in the first message
        chat_history = []
        
        # Add the image and initial prompt as the first turn
        if "uploaded_file_content" in st.session_state:
             image_part = {
                "mime_type": st.session_state.uploaded_file_type,
                "data": st.session_state.uploaded_file_content
            }
             chat_history.append({
                 "role": "user",
                 "parts": [image_part, "Analyze this medical image based on the system instructions."]
             })
        
        # Add the rest of the history
        # Note: st.session_state.history stores simple text parts for display. 
        # We need to be careful not to duplicate the first turn if we already added it.
        # Let's rebuild the history from st.session_state.history, skipping the first implied turn if we added it manually.
        
        for msg in st.session_state.history:
             if msg["parts"][0] == "Analyze this medical image.":
                 # We already added the "real" first turn with the image above.
                 # The model response to this is also in history, we need to add that.
                 continue
             
             chat_history.append(msg)

        # Add the new user message
        chat_history.append({"role": "user", "parts": [prompt]})
        
        # Generate response
        try:
            # We can use a chat session or just generate_content with the full list
            # generate_content is stateless, so we pass the full history
            
            # Convert chat_history to the format expected by generate_content (list of contents)
            # Content objects or dicts. The dicts we built above are compatible.
            
            with st.spinner("Thinking..."):
                response = model.generate_content(chat_history)
                
                # Assisstant ka response show krte rehna hai 
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                
                # History update krte rehna hai 
                st.session_state.history.append({"role": "user", "parts": [prompt]})
                st.session_state.history.append({"role": "model", "parts": [response.text]})
                
        except Exception as e:
            st.error(f"An error occurred during follow-up: {e}")
