from flask import Flask,request,jsonify
import joblib

app = Flask(__name__)


@app.route('/py_server')
def index():
    return 'Running Python Flask Server'

model = joblib.load('./model.pkl')
encoder = joblib.load('./encoder.pkl')

@app.route('/api/cancer_prediction',methods=['POST'])
def cancer_prediction_api():
    content = request.json
    results = cancer_prediction(content,model,encoder)
    return str(results[0])
    # return jsonify(results)

def cancer_prediction(data,model,encoder):
    brca1_chromosome = 17
    brca1_reference = data['brca1_ref']
    brca1_alternate = data['brca1_alt']
    brca2_chromosome = 13
    brca2_reference = data['brca2_ref']
    brca2_alternate = data['brca2_alt']
    atm_chromosome = 11
    atm_reference = data['atm_ref']
    atm_alternate = data['atm_alt']

    
    prediction_data = [[
        brca1_chromosome,
        brca1_reference,
        brca1_alternate,
        brca2_chromosome,
        brca2_reference,
        brca2_alternate,
        atm_chromosome,
        atm_reference,
        atm_alternate
    ]]
    
    
    encoded_data = encoder.transform(prediction_data)
    
    prediction = model.predict(encoded_data)
    
    return prediction



if __name__ == '__main__':
    app.run()