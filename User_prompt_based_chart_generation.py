import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# List of chart types that the model can predict
chart_labels = [
    "histogram", "scatter", "lineplot", "bar", "boxplot", "heatmap", "piechart", 
    "areachart", "violin", "sunburst", "treemap", "funnel", "density_heatmap", "density_contour", "clustered_column"
]

def generate_chart(action, df, highlight_values=None):
    """Generates charts based on the specified action and data."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Check if there are numerical columns
    if numerical_cols.empty:
        st.write("No numerical columns available for chart generation.")
        return

    # Generate charts based on the action input (e.g., histogram, scatter, etc.)
    if action == "histogram" and len(numerical_cols) > 0:
        fig = px.histogram(df, x=numerical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Histogram")
        st.plotly_chart(fig)
    elif action == "scatter" and len(numerical_cols) > 1:
        fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1])
        add_highlights(fig, highlight_values, scatter=True)
        fig.update_layout(title="Scatter Plot")
        st.plotly_chart(fig)
    elif action == "lineplot" and len(numerical_cols) > 1:
        fig = px.line(df, x=numerical_cols[0], y=numerical_cols[1])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Line Plot")
        st.plotly_chart(fig)
    elif action == "bar" and len(categorical_cols) > 0 and len(numerical_cols) > 0:
        fig = px.bar(df, x=categorical_cols[0], y=numerical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Bar Plot")
        st.plotly_chart(fig)
    elif action == "boxplot" and len(numerical_cols) > 0:
        fig = px.box(df, y=numerical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Box Plot")
        st.plotly_chart(fig)
    elif action == "heatmap" and len(numerical_cols) > 1:
        fig = px.imshow(df.corr(), text_auto=True)
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Heatmap")
        st.plotly_chart(fig)
    elif action == "piechart" and len(categorical_cols) > 0:
        fig = px.pie(df, names=categorical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Pie Chart")
        st.plotly_chart(fig)
    elif action == "areachart" and len(numerical_cols) > 0:
        fig = px.area(df, x=numerical_cols[0], y=numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Area Chart")
        st.plotly_chart(fig)
    elif action == "violin" and len(numerical_cols) > 0:
        fig = px.violin(df, y=numerical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Violin Plot")
        st.plotly_chart(fig)
    elif action == "sunburst" and len(categorical_cols) > 1:
        fig = px.sunburst(df, path=[categorical_cols[0], categorical_cols[1]])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Sunburst Chart")
        st.plotly_chart(fig)
    elif action == "treemap" and len(categorical_cols) > 1:
        fig = px.treemap(df, path=[categorical_cols[0], categorical_cols[1]])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Treemap")
        st.plotly_chart(fig)
    elif action == "funnel" and len(categorical_cols) > 0 and len(numerical_cols) > 0:
        fig = px.funnel(df, x=categorical_cols[0], y=numerical_cols[0])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Funnel Chart")
        st.plotly_chart(fig)
    elif action == "density_heatmap" and len(numerical_cols) > 1:
        fig = px.density_heatmap(df, x=numerical_cols[0], y=numerical_cols[1])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Density Heatmap")
        st.plotly_chart(fig)
    elif action == "density_contour" and len(numerical_cols) > 1:
        fig = px.density_contour(df, x=numerical_cols[0], y=numerical_cols[1])
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Density Contour")
        st.plotly_chart(fig)
    elif action == "clustered_column" and len(categorical_cols) > 0 and len(numerical_cols) > 1:
        fig = px.bar(df, x=categorical_cols[0], y=numerical_cols)
        add_highlights(fig, highlight_values)
        fig.update_layout(title="Clustered Column Chart")
        st.plotly_chart(fig)
    else:
        st.write(f"No chart generation logic for action: {action}")

def add_highlights(fig, highlight_values, scatter=False):
    """Adds highlights (e.g., avg, min, max) to charts."""
    if highlight_values:
        if scatter:
            # For scatter plots, highlight avg, min, max points
            fig.add_scatter(x=[highlight_values['avg'][0]], y=[highlight_values['avg'][1]], mode='markers', marker=dict(color='red', size=10), name="Avg")
            fig.add_scatter(x=[highlight_values['max'][0]], y=[highlight_values['max'][1]], mode='markers', marker=dict(color='green', size=10), name="Max")
            fig.add_scatter(x=[highlight_values['min'][0]], y=[highlight_values['min'][1]], mode='markers', marker=dict(color='blue', size=10), name="Min")
        else:
            # For other plots, highlight with vertical lines
            fig.add_vline(x=highlight_values['avg'], line=dict(color="red", dash="dash"), annotation_text="Avg")
            fig.add_vline(x=highlight_values['max'], line=dict(color="green", dash="dash"), annotation_text="Max")
            fig.add_vline(x=highlight_values['min'], line=dict(color="blue", dash="dash"), annotation_text="Min")

def preprocess_text(text):
    """Preprocesses input text by removing stop words and punctuation."""
    doc = nlp(text.lower())
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

class ChartDataset(Dataset):
    """Custom dataset for handling text data for model training."""
    def __init__(self, prompts, labels, vocab, max_len=50):
        self.prompts = prompts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        tokens = preprocess_text(prompt)
        indexed_tokens = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        indexed_tokens = indexed_tokens[:self.max_len] + [self.vocab["<PAD>"]] * (self.max_len - len(indexed_tokens))
        return torch.tensor(indexed_tokens), torch.tensor(label)

class TextClassificationModel(nn.Module):
    """Bidirectional LSTM model for text classification."""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=15, dropout=0.5):  # Fix output_dim to 15
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Corrected output dimension to 15

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        final_hidden_state = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate final states from both directions
        out = self.fc(self.dropout(final_hidden_state))
        return out

def prepare_dataset():
    """Prepares dataset for training."""
    prompts = [
        "show me a histogram of sales", "scatter plot of sales vs time", "plot a line chart for revenue", 
        "bar chart of categories", "boxplot of age distribution", "heatmap of correlation", 
        "pie chart for market share", "area chart for sales trends", "violin plot of prices", 
        "sunburst chart for hierarchy", "treemap for expenditures", "funnel chart for conversion rates", 
        "density heatmap of data", "density contour for variable distribution", "clustered column chart"
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Corrected labels, 0 to 14
    all_tokens = [token for prompt in prompts for token in preprocess_text(prompt)]
    vocab = {token: idx + 1 for idx, token in enumerate(set(all_tokens))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)
    train_prompts, test_prompts, train_labels, test_labels = train_test_split(prompts, labels, test_size=0.2)
    train_dataset = ChartDataset(train_prompts, train_labels, vocab)
    test_dataset = ChartDataset(test_prompts, test_labels, vocab)
    return train_dataset, test_dataset, vocab

def train_model(train_dataset, vocab_size):
    """Trains the model using the training dataset."""
    model = TextClassificationModel(vocab_size)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model.train()
    for epoch in range(5):  # Train for 5 epochs
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def main():
    # Streamlit interface to upload dataset and interact with the model
    import streamlit as st

    # Title of the app
    st.title("Data Visualization Tool")

    # Instructions for the user
    st.markdown("""
        ## Instructions
        
        Welcome to the **Data Visualization Tool**!

        This app allows you to upload a dataset (in CSV format) and generate various types of visualizations based on your input prompt.

        ### Steps to use:
        1. **Upload Your Dataset**: Choose a CSV file containing your data using the "Upload your dataset" button.
        2. **Enter a Visualization Prompt**: Once the dataset is uploaded, type a prompt like "show me a line plot of amount vs date" to specify which chart you want to generate.
        3. **View the Chart**: The app will automatically generate the chart based on your prompt and display it on the page.

        ### Supported Visualizations:
        - Line Plot
        - Bar Chart
        - Scatter Plot
        - Histogram
        - Box Plot
        - Heatmap
        - Pie Chart
        - Area Chart
        - Violin Plot
        - Sunburst
        - Treemap
        - Funnel Chart
        - Density Heatmap
        - Clustered Column Chart

        If you're not sure what to enter, try something like:
        - "show me a bar chart of sales vs date"
        - "give me a pie chart of categories"
        - "display a scatter plot of price vs quantity"

        Happy visualizing!
    """)

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        imputer = SimpleImputer(strategy='mean')
        df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df[df.select_dtypes(include=['float64', 'int64']).columns])
        st.write(df.head())
        train_dataset, test_dataset, vocab = prepare_dataset()
        model = train_model(train_dataset, len(vocab))
        
        new_prompt = st.text_input("Enter a prompt for chart prediction", "show me a clustered column chart for sales")
        if new_prompt:
            model.eval()
            new_input = torch.tensor([vocab.get(token, vocab["<UNK>"]) for token in preprocess_text(new_prompt)])
            new_input = new_input.unsqueeze(0).to(device)  # Add batch dimension
            prediction = model(new_input)
            predicted_label = prediction.argmax(dim=1).item()
            
            st.write(f"The predicted chart type is: {chart_labels[predicted_label]}")
            generate_chart(chart_labels[predicted_label], df)

if __name__ == "__main__":
    main()
