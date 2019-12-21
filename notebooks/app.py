import numpy as np
from flask Import Flask, request, jsonify, render_template

app = Flask(__name__)

filename = 'model1.pickle'
model = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=[POST])
def predict():
#     for rendering results in HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text = 'Risk of this patient to have a heart disease is $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)