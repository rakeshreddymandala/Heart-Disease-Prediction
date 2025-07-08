# Heart Disease Prediction Web App

This project is a web application for predicting heart disease using a trained machine learning model. It uses Flask for the backend and HTML/CSS for the frontend.

## Features

- User-friendly form for inputting patient data
- Machine learning model for prediction (using `our_model.pkl`)
- Responsive and modern UI

---

## Folder Structure

```
documents/
│
├── app.py
├── our_model.pkl
├── static/
│   ├── style.css
│   └── result.css
└── templates/
    ├── home.html
    └── after.html
```

---

## Requirements

- Python 3.x
- Flask
- numpy
- pickle (standard library)
- The file `our_model.pkl` (your trained model)

Install dependencies:
```bash
pip install flask numpy
```

---

## Running the Backend

1. Make sure `app.py` and `our_model.pkl` are in the same directory.
2. Run the Flask app:
    ```bash
    python app.py
    ```
3. The server will start at `http://127.0.0.1:5000/` by default.

---

## Using the Frontend

- Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- Fill out the form and submit to get a prediction.
- The result page will display the prediction and a button to return to the home page.

---

## Notes

- All static files (CSS) are in the `static` folder.
- All HTML templates are in the `templates` folder.
- If you update the model, replace `our_model.pkl` with your new model file.

---

## Troubleshooting

- If styles are not loading, ensure the `static` folder is in the same directory as `app.py` and the file names match.
- If you get a "file not found" error for `our_model.pkl`, make sure it exists in the project root.

---

## License

This project is for educational purposes.
