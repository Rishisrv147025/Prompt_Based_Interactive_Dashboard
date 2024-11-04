import os
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK resources (uncomment the following lines if running for the first time)
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Custom LSTM Model for NLP
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, 50)  # Embedding layer
        self.lstm = nn.LSTM(50, hidden_size, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.embedding(x)  # Pass through embedding layer
        x, (hn, cn) = self.lstm(x)  # Pass through LSTM layer
        x = self.fc(hn[-1])  # Use the last hidden state
        return x

# Load and prepare user prompts dataset
def load_prompt_data():
    prompts = [
        "Show me a histogram", 
        "Display a scatter plot", 
        "Generate a line chart", 
        "Create a bar chart", 
        "Show me the pie chart", 
        "Display the heatmap", 
        "Create a box plot",
        "Show summary statistics",
        "Show me a correlation matrix",
        "What is the average of", 
        "What is the mean of", 
        "What is the median of", 
        "What is the mode of", 
        "What is the standard deviation of"
    ]
    actions = [
        "histogram", 
        "scatter", 
        "line", 
        "bar", 
        "pie", 
        "heatmap", 
        "box", 
        "summary", 
        "correlation",
        "average", 
        "mean", 
        "median", 
        "mode", 
        "std_deviation"
    ]
    return prompts, actions

# Function to train the LSTM model
def train_nlp_model(prompts, actions):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(actions)

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(prompts)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized.toarray(), y_encoded, test_size=0.2, random_state=42)

    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)

    hidden_size = 64
    model = LSTMClassifier(input_size=len(vectorizer.vocabulary_), hidden_size=hidden_size, num_classes=len(label_encoder.classes_))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, vectorizer, label_encoder

# Global variables for the model and analysis results
nlp_model = None
vectorizer = None
label_encoder = None
cleaned_df = None
univariate_analysis = None
bivariate_analysis = None

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global cleaned_df, univariate_analysis, bivariate_analysis
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        cleaned_df = clean_data(df)
        univariate_analysis = perform_univariate_analysis(cleaned_df)
        bivariate_analysis = perform_bivariate_analysis(cleaned_df)

        column_names = cleaned_df.columns.tolist()

        return render_template('analysis.html', 
                               univariate_analysis=univariate_analysis,
                               bivariate_analysis=bivariate_analysis,
                               filename=filename,
                               column_names=column_names,
                               error=None)  # Initialize error as None
    else:
        return redirect(request.url)

# Function for data cleaning
def clean_data(df):
    df.fillna(method='ffill', inplace=True)
    return df

# Function for univariate analysis
def perform_univariate_analysis(df):
    summary = df.describe()
    hist_fig = px.histogram(df, x=df.columns[0])
    return summary, hist_fig.to_html(full_html=False)

# Function for bivariate analysis
def perform_bivariate_analysis(df, x_col=None, y_col=None):
    if x_col is None or y_col is None:
        return None

    if df.shape[1] >= 2:
        scatter_fig = px.scatter(df, x=x_col, y=y_col)
        return scatter_fig.to_html(full_html=False)
    return None

# Function to calculate statistics
def calculate_statistics(column):
    stats = {
        "mean": column.mean(),
        "median": column.median(),
        "mode": column.mode()[0] if not column.mode().empty else None,
        "std_deviation": column.std(),
        "average": column.mean()
    }
    return stats

# Function to preprocess user input using NLTK
def preprocess_input(user_input):
    # Tokenize the input
    tokens = word_tokenize(user_input.lower())
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

# Route to handle user prompts and chart updates
@app.route('/update_chart', methods=['POST'])
def update_chart():
    global cleaned_df, univariate_analysis, bivariate_analysis
    user_input = request.form.get('user_input')
    processed_input = preprocess_input(user_input)  # Preprocess the user input
    action = interpret_prompt(processed_input)  # Interpret the processed input

    selected_x = request.form.get('x_column')
    selected_y = request.form.get('y_column')

    error_message = None  # Initialize an error message variable

    try:
        if action == "correlation":
            correlation_matrix = cleaned_df.corr()
            corr_fig = px.imshow(correlation_matrix, color_continuous_scale='Viridis')
            bivariate_analysis = corr_fig.to_html(full_html=False)

        elif action in ["average", "mean", "median", "mode", "std_deviation"]:
            if selected_x and cleaned_df[selected_x].dtype in ['int64', 'float64']:
                stats = calculate_statistics(cleaned_df[selected_x])
                result = f"For column '{selected_x}': "
                result += ', '.join([f"{key.capitalize()}: {value}" for key, value in stats.items() if value is not None])
                error_message = result  # Use error_message to display statistics
            else:
                error_message = f"Column '{selected_x}' must be numerical for statistics."

        elif action == "histogram":
            if cleaned_df[selected_x].dtype in ['int64', 'float64']:
                hist_fig = px.histogram(cleaned_df, x=selected_x)
                bivariate_analysis = hist_fig.to_html(full_html=False)
            else:
                error_message = 'Selected column for histogram must be numerical.'

        elif action == "scatter":
            if cleaned_df[selected_x].dtype in ['int64', 'float64'] and cleaned_df[selected_y].dtype in ['int64', 'float64']:
                scatter_fig = px.scatter(cleaned_df, x=selected_x, y=selected_y)
                scatter_fig.add_annotation(x=cleaned_df[selected_x].mean(),
                                           y=cleaned_df[selected_y].mean(),
                text="Mean", showarrow=True, arrowhead=1)
                bivariate_analysis = scatter_fig.to_html(full_html=False)
            else:
                error_message = 'Both selected columns for scatter plot must be numerical.'

        elif action == "line":
            if cleaned_df[selected_x].dtype in ['int64', 'float64'] and cleaned_df[selected_y].dtype in ['int64', 'float64']:
                line_fig = px.line(cleaned_df, x=selected_x, y=selected_y)
                line_fig.add_annotation(x=cleaned_df[selected_x].mean(), 
                                        y=cleaned_df[selected_y].mean(),
                                        text="Mean", showarrow=True, arrowhead=1)
                bivariate_analysis = line_fig.to_html(full_html=False)
            else:
                error_message = 'Both selected columns for line chart must be numerical.'

        elif action == "bar":
            if cleaned_df[selected_x].dtype == 'object' and cleaned_df[selected_y].dtype in ['int64', 'float64']:
                bar_fig = px.bar(cleaned_df, x=selected_x, y=selected_y)
                bar_fig.add_annotation(x=cleaned_df[selected_x].mode()[0], 
                                       y=cleaned_df[selected_y].mean(),
                                       text="Average", showarrow=True, arrowhead=1)
                bivariate_analysis = bar_fig.to_html(full_html=False)
            else:
                error_message = 'X column must be categorical and Y column must be numerical for bar chart.'

        elif action == "box":
            if cleaned_df[selected_x].dtype in ['int64', 'float64']:
                box_fig = px.box(cleaned_df, y=selected_x)
                box_fig.add_annotation(y=cleaned_df[selected_x].mean(),
                                       text="Mean", showarrow=True, arrowhead=1)
                bivariate_analysis = box_fig.to_html(full_html=False)
            else:
                error_message = 'Selected column for box plot must be numerical.'

        elif action == "heatmap":
            correlation_matrix = cleaned_df.corr()
            corr_fig = px.imshow(correlation_matrix, color_continuous_scale='Viridis')
            bivariate_analysis = corr_fig.to_html(full_html=False)

        else:
            error_message = f'Error: "{user_input}" is not a recognized command. Please try again with a valid command!'

    except Exception as e:
        error_message = f'An error occurred while generating the chart: {str(e)}'

    return render_template('analysis.html', 
                           univariate_analysis=univariate_analysis,
                           bivariate_analysis=bivariate_analysis,
                           filename="uploaded_file.csv",  # Replace with actual filename if needed
                           column_names=cleaned_df.columns.tolist(),
                           error=error_message)  # Pass the error message to the template

# Function to interpret user prompts and return corresponding actions
def interpret_prompt(prompt):
    global nlp_model, vectorizer, label_encoder

    prompt_vectorized = vectorizer.transform([prompt]).toarray()
    input_tensor = torch.LongTensor(prompt_vectorized)

    with torch.no_grad():
        predictions = nlp_model(input_tensor)
    
    _, predicted = torch.max(predictions, 1)
    action = label_encoder.inverse_transform(predicted.numpy())[0]
    return action

# Load prompt data and train model on startup
prompts, actions = load_prompt_data()
nlp_model, vectorizer, label_encoder = train_nlp_model(prompts, actions)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
