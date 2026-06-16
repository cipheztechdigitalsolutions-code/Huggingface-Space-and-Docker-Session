title: Iris Flower Classifier
emoji: 🌸
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---




# Iris Flower Classifier 🌸


A simple ML model deployed with **Gradio + Docker** on **Hugging Face Spaces**.


Adjust the four flower measurements and the model predicts the iris species
(setosa, versicolor, or virginica).


## Stack
- **Vs Code**
- **scikit-learn** — RandomForest model
- **Gradio** — web interface
- **Docker** — containerization
- **Hugging Face Spaces** — free hosting


## Run locally
```bash
pip install -r requirements.txt
python train.py     # creates model.pkl + labels.pkl
python app.py       # opens at http://localhost:7860
```

```python
name = "Mirack"
```

## Run with Docker
```bash
docker build -t iris-app .
docker run -p 7860:7860 iris-app
```
