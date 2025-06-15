# ESRS-Assistant: an AI-based chatbot

ESRS-Assistant is a chatbot designed to query the official ESRS documentation efficiently and accurately. It answers user questions based solely on the official documents, and politely declines to answer when the information is not available.

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

## Usage

1. Run the chatbot: (in the code folder)
   ```sh
   cd code
   streamlit run app.py
   ```

## Folder and code introduction

### /scraping_and tagging
1. scrap_text: the result of scarping from the html https://xbrl.efrag.org/e-esrs/esrs-set1-2023.html#d1e5302-3-1 
2. tagged_text: the result of tagging from the html. the json is used by the app.py.
3. scraping.ipynb: the code to scrap the text from the html paragraph by paragraph according to the html structure.
4. tagging.ipynb: the code to add metadata to the scrapped text.

### /code
1. app.py includes all the code of the RAG architecture.
2. .streamlit/ contains the configuration to run the app locally while using the LLM hosted by Databricks.

### /checkpoints
1. faiss_index: store the vectorbase for the sake of faster running from the second time.
2. all_text: store all the text of the ESRS documents. This file is normally not useful

###/dataset
This folder includes 4 docuemnts, which are the main ESRS documents and three FAQs from the official website.

### /img_icon
It includes the icon image we use for the streamlit app.


## Deployment

It is possible to deploy this app on the Databricks platform after revising some configurations.




