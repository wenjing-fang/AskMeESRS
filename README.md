# Demo with Evaluation Function  

This project provides a chatbot with an evaluation function, allowing users to interact with the chatbot and assess its performance using a predefined dataset.

## Installation  

1. Clone this repository:  
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Start the Ollama model:
   ```sh
   ollama run llama3.2
   ```

## Usage

1. Run the chatbot: (in the code folder)
   ```sh
   streamlit run demo.py
   ```

2. Evaluate the chatbot: (in the code folder)   
   ```sh
   streamlit run test_evaluate.py
   ```

3. The evaluation results will be saved in evaluation/evaluation_result.csv and evaluation/evaluation_result.json

