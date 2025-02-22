import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import NameDataset

# Load dataset
@st.cache_resource
def load_dataset():
    dataset = NameDataset(r"C:\Users\user\OneDrive\Desktop\Resources\Selected Topics in AI\Assignment1\names")
    return dataset

dataset = load_dataset()

# Define Model
class NameClassifier(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size):
        super(NameClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 20, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.size(0), -1)
        hidden = self.fc1(embedded)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output

# Load Model
@st.cache_resource
def load_model():
    model = NameClassifier(
        input_size=len(dataset.all_letters),
        embedding_dim=128,
        hidden_size=256,
        output_size=len(dataset.nationality_to_ix)
    )
    model_path = r"C:\Users\user\OneDrive\Desktop\Resources\Selected Topics in AI\Assignment1\model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict_nationality(name, top_k=3):
    """Predict nationality from a name."""
    with torch.no_grad():
        name_tensor = dataset.name_to_tensor(name.lower()).unsqueeze(0)
        output = model(name_tensor)
        probabilities = F.softmax(output, dim=1)
        top_prob, top_idx = torch.topk(probabilities, top_k)

        results = []
        ix_to_nationality = {v: k for k, v in dataset.nationality_to_ix.items()}
        for prob, idx in zip(top_prob[0], top_idx[0]):
            nationality = ix_to_nationality[idx.item()]
            results.append((prob.item(), nationality))
    
    return results

# Streamlit UI
st.markdown(
    """
    <h1 style="text-align: center; font-size: 3em; color: #581845;">
        🌍 Nationality Predictor
    </h1>
    <h3 style="text-align: center; color: #581845;">Enter a name and discover its possible origins!</h3>
    <hr style="border:1px solid #581845;">
    """,
    unsafe_allow_html=True
)

# Center input field
name = st.text_input("🔤 **Type a name below:**", placeholder="e.g., Tasneem, Ivan, Mohammed")

# Flag mapping (add more as needed)
flag_map = {
    "French": "🇫🇷", "Italian": "🇮🇹", "Spanish": "🇪🇸", "German": "🇩🇪", 
    "Russian": "🇷🇺", "Japanese": "🇯🇵", "Chinese": "🇨🇳", "Indian": "🇮🇳", 
    "English": "🇬🇧", "Arabic": "🇸🇦", "Turkish": "🇹🇷", "Greek": "🇬🇷"
}

if name:
    predictions = predict_nationality(name)
    st.subheader(f"🔍 Predictions for '{name}':")

    for prob, nationality in predictions:
        flag = flag_map.get(nationality, "🏳") 
        st.markdown(
            f"""
            <div style="
                padding: 10px; 
                margin: 10px 0; 
                background-color: #f9f9f9; 
                border-radius: 10px; 
                border-left: 6px solid #581845;
            ">
                <h3 style="display: flex; align-items: center;">
                    {flag} {nationality} <span style="margin-left: auto; color: #555;">{prob:.2%}</span>
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
