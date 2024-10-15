# Import necessary libraries
import google.generativeai as genai
import streamlit as st
import os
import joblib
import plotly.graph_objects as go
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Please provide the Google API key in the .env file.")
    st.stop()

# Configure the API key for Google Generative AI
genai.configure(api_key=google_api_key)

# App title and description
st.title("💬 EmotionAI Chat")
st.caption("🚀 A chatbot that integrates emotion recognition powered by AI.")

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Select Emotion Classifier Model", options=["Random Forest", "Logistic Regression"]
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


# Function to display chat messages
def display_chat_history():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


# Function to generate a response using Google Generative AI
def get_ai_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        st.error(f"Error with Google Generative AI: {str(e)}")
        return "Sorry, I couldn't process that."


# Function to load the emotion classifier model
def load_emotion_model(model_name):
    try:
        model_file = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        model = joblib.load(
            os.path.join(r"C:\Users\Aneesh Angane\Desktop\test", model_file)
        )
        return model
    except FileNotFoundError:
        st.error("Emotion classifier model file not found.")
        return None


# Function to generate emotion confidence charts
def plot_emotion_charts(probabilities, class_names):
    colors = sns.color_palette("husl", len(class_names))

    # Plotly bar chart
    bar_fig = go.Figure()
    for i, (emotion, prob) in enumerate(zip(class_names, probabilities[0])):
        bar_fig.add_trace(
            go.Bar(
                x=[emotion],
                y=[prob * 100],  # Convert to percentages
                name=emotion,
                marker_color=colors[i],
            )
        )
    bar_fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis_title="Emotion",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),  # Limit y-axis to 0-100%
        xaxis_tickangle=-45,
    )

    # Plotly pie chart
    pie_fig = go.Figure(
        go.Pie(
            labels=class_names,
            values=[prob * 100 for prob in probabilities[0]],
            hoverinfo="label+percent",
            textinfo="label+percent",
            marker=dict(colors=colors),
        )
    )
    pie_fig.update_layout(title="Emotion Distribution")

    # Display the charts
    st.plotly_chart(bar_fig)
    st.plotly_chart(pie_fig)


# Display chat history
display_chat_history()

# User input chat box
user_input = st.chat_input()

# If user submits input, process the input and generate a response
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI response
    ai_response = get_ai_response(user_input)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)

    # Save last input for emotion analysis
    st.session_state["last_input"] = user_input

# Button to generate emotion report based on the last user input
if st.button("Get Emotion Report") and "last_input" in st.session_state:
    emotion_model = load_emotion_model(model_option)

    if emotion_model:
        # Predict emotion probabilities
        try:
            last_input = st.session_state["last_input"]
            probabilities = emotion_model.predict_proba([last_input])
            class_names = emotion_model.classes_

            # Plot the emotion charts
            plot_emotion_charts(probabilities, class_names)
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
