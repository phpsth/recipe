import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random


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
st.title("< AI-Powered Menu Suggestion />")
st.write("Enter the ingredients you have, and I will suggest a menu name")

# Initialize session state for ingredients if it doesn't exist
if "ingredients_input" not in st.session_state:
    st.session_state.ingredients_input = ""

# Common ingredients as buttons
st.write("Click to add ingredients:")
common_ingredients = [
    "chicken",
    "beef",
    "pork",
    "fish",
    "eggs",
    "flour",
    "sugar",
    "salt",
    "pepper",
    "onion",
    "garlic",
    "milk",
    "butter",
    "oil",
    "rice",
    "pasta",
    "bread",
    "cheese",
    "tomatoes",
    "potatoes",
]
cols = st.columns(5)
for i, ingredient in enumerate(
    common_ingredients[:10]
):  # Displaying first 10 as buttons
    if cols[i % 5].button(ingredient, key=f"common_{ingredient}"):
        if st.session_state.ingredients_input:
            st.session_state.ingredients_input += f", {ingredient}"
        else:
            st.session_state.ingredients_input = ingredient
        st.rerun()  # Use st.rerun() to update the text input


# Text input for ingredients
user_input = st.text_input(
    "Or just type here the ingredients (comma-separated):",
    value=st.session_state.ingredients_input,
    key="ingredient_textbox",
)

# Update session state if text input changes
st.session_state.ingredients_input = user_input

# Generate button
if st.button("Show me magic_"):
    if st.session_state.ingredients_input:
        ingredients_list = [
            item.strip()
            for item in st.session_state.ingredients_input.split(",")
            if item.strip()
        ]
        if ingredients_list:
            with st.spinner("Generating menu name..."):
                generated_name = generate_menu_name(ingredients_list, model, tokenizer)
                st.success(f"Suggested Menu Name: {generated_name}")
        else:
            st.warning("Please enter some ingredients.")
    else:
        st.warning("Please enter or select some ingredients.")

# Optional: Clear selected ingredients button
if st.button("Clear Ingredients"):
    st.session_state.ingredients_input = ""
    st.rerun()  # Use st.rerun() to clear the text input
