import pickle

class CardioRiskPredictor:
    def __init__(self, model_file='model.pkl'):
        with open(model_file, 'rb') as f:
            self.pickled_model = pickle.load(f)

    def predict(self, age, gender, height, weight, ap_hi, ap_low, chol, gluc, smoke, alco, active):
        """
        Age should be int between 1-100
        Gender should be int 1 (Male) or 2 (Female)
        Height should be between 50-280 cm
        Weight should be between 15-125 kg
        Systolic Blood Pressure should be between 100-200 mm Hg
        Diastolic Blood Pressure should be between 70-130 mm Hg
        Cholesterol has three levels, 1-3. Should be a drop down menu
        Glucose levels has three levels, 1-3. Should be a drop down menu
        Smoke, Alcohol, and Active should be are 1 (yes) or 0 (no)
        """
        array = [[age, gender, height, weight, ap_hi, ap_low, chol, gluc, smoke, alco, active]]
        per = self.pickled_model.predict_proba(array)[:, 1][0]
        return round(per * 100, 2)
