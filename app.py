import joblib
import gradio as gr

model = joblib.load("model.pkl")
labels = joblib.load("labels.pkl")


def predict(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    probs = model.predict_proba(features)[0]

    return {labels[i]: float(probs[i]) for i in range(len(labels))}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.8, label="Sepal Length (cm)"),
        gr.Slider(2.0, 5.0, value=2.8, label="Sepal Width (cm)"),
        gr.Slider(1.0, 8.0, value=3.8, label="Petal Length (cm)"),
        gr.Slider(1.0, 3.0, value=1.5, label="Petal Width (cm)")
    ], 
    outputs=gr.Label(num_top_classes=2, label="Prediction"), 
    title="Iris Flower Classifier",
    description="Adjust the sliers and the model predicts the iris species, Built Gradio on Docker, and deployed Hugging Face Spaces"

)


if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860)
