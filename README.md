# Hotel Assistant Chatbot
<p align="center">
  <img src="https://github.com/FOX2920/hotel_chatbot/assets/91709267/9a104692-1a68-44d4-a52c-e971f906d3c2" alt="Pineline">
</p>


This project implements a Hotel Assistant Chatbot using Flask, Google Generative AI Embeddings, FAISS, and LangChain. The chatbot provides information about hotel rooms and restaurants based on data from CSV files.

## Prerequisites

- Anaconda/Miniconda installed
- Python 3.10.14
- Necessary CSV files: `room.csv` and `restaurant.csv`
- Gemini API key: Get it from [Google AI Studio](https://ai.google.dev/tutorials/setup?hl=tr)
- Create a `.env` file with your Gemini API key and other necessary environment variables

## Setup

### Step 1: Create a Conda Environment

1. Open your terminal or Anaconda Prompt.
2. Create a new Conda environment with Python 3.10.14:

   ```sh
   conda create -n hotel-assistant python=3.10.14
   ```

### Step 2: Activate the Environment

   ```sh
   conda activate hotel-assistant
   ```

### Step 3: Install Required Packages

   ```sh
   pip install -r requirements.txt
   ```

### Step 4: Prepare CSV Files: Ensure you have the `room.csv` and `restaurant.csv` files in the project directory.

### Step 5: Set Up Environment Variables

  Create a `.env` file in the project directory with the following content:

   ```dotenv
   GOOGLE_API_KEY="YOUR_API_KEY"
   ```

## Running the Chatbot

### Step 1: Start the Flask Application

1. Run the Flask app by executing the following command in your terminal:

   ```sh
   cd path/to/hotel_chatbot
   python api.py
   ```

2. The application will start, and you should see output indicating that the server is running, for example:

   ```sh
   * Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)
   ```

### Step 2: Interact with the Chatbot

1. Open Postman.

2. Set the request type to POST.

3. Enter the request URL: http://127.0.0.1:8080/chat

4. In the Headers section, add a new header:
```sh
    + Key: Content-Type
    + Value: application/json
```
5. In the Body section, select the raw option and choose JSON (application/json) from the dropdown menu.

6. Enter your JSON request body in the text area. For example:

```json
{
    "question": "Tôi có gia đình 4 người thì nên thuê phòng nào?"
}
```
7. Click the Send button.You should receive a response from the chatbot in the Response Body section of Postman.
 
<p align="center">
  <img src="https://github.com/FOX2920/hotel_chatbot/assets/91709267/202adf07-f7ae-49c2-8488-2389c5a82f45" alt="Test">
</p>

## Project Structure

- `api.py`: Main Flask application file.
- `requirements.txt`: List of required Python packages.
- `room.csv`: CSV file containing information about hotel rooms.
- `restaurant.csv`: CSV file containing information about hotel restaurants.
- `.env`: Environment variables file.

## Requirements

The `requirements.txt` file should include the following dependencies:

```text
Flask
pandas
langchain
langchain-google-genai
langchain-community
faiss-cpu
python-dotenv
```

## Acknowledgments

This project uses the following libraries and services:

- [Flask](https://flask.palletsprojects.com/)
- [Pandas](https://pandas.pydata.org/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Google Generative AI Embeddings](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

Feel free to reach out if you have any questions or need further assistance!
