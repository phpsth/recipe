import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


@st.cache_resource  # Cache the model and tokenizer
def load_model():
    """Loads the fine-tuned T5 model and tokenizer."""
    model_path = "./fine_tuned_t5_recipe_model"  # Path to your saved model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model


tokenizer, model = load_model()


def generate_menu_name(ingredients, model, tokenizer, max_length=50, num_beams=5):
    """Generates a menu name based on a list of ingredients."""
    input_text = ", ".join(ingredients)
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    # Ensure the model is on the correct device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)

    output_ids = model.generate(
        input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# Streamlit app interface
st.title("< AI-powered Menu Suggestion />")
st.write("Enter the ingredients you have, and I will suggest a menu name!")

user_input = st.text_input("Ingredients (comma-separated):", "")

if user_input:
    ingredients_list = [item.strip() for item in user_input.split(",") if item.strip()]
    if ingredients_list:
        with st.spinner("Generating menu name..."):
            generated_name = generate_menu_name(ingredients_list, model, tokenizer)
            st.success(f"Suggested Menu Name: {generated_name}")
    else:
        st.warning("Please enter some ingredients.")
