import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_text(keywords, genre, text_type, max_length=100, temperature=0.7):
    """Generates a story or poem based on user inputs."""
    prompt = f"A {genre} {text_type} about {', '.join(keywords)}: "

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("AI Story & Poem Generator")
st.markdown("Generate creative text based on your **keywords** and **genre**!")

# User Inputs
keywords = st.text_input("Enter keywords (comma-separated)", "moon, adventure, magic")
genre = st.selectbox("Choose a Genre", ["Fantasy", "Horror", "Romance", "Sci-Fi", "Adventure"])
text_type = st.radio("Generate a:", ["Story", "Poem"])
max_length = st.slider("Text Length", min_value=50, max_value=300, value=100)
temperature = st.slider("Creativity (Temperature)", min_value=0.5, max_value=1.5, value=0.7)

if st.button("Generate Text ✨"):
    with st.spinner("Generating... Please wait "):
        keywords_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        if not keywords_list:
            st.error("Please enter at least one keyword!")
        else:
            generated_text = generate_text(keywords_list, genre.lower(), text_type.lower(), max_length, temperature)
            st.subheader("Generated Text:")
            st.write(generated_text)
