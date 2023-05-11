from django.shortcuts import render
from modelTraining.main import CardioRiskPredictor

def home(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        gender = int(request.POST.get('gender'))
        height = float(request.POST.get('height'))
        weight = float(request.POST.get('weight'))
        ap_hi = int(request.POST.get('ap_hi'))
        ap_lo = int(request.POST.get('ap_lo'))
        cholesterol = int(request.POST.get('cholesterol'))
        gluc = int(request.POST.get('gluc'))
        smoke = int(request.POST.get('smoke'))
        alco = int(request.POST.get('alco'))
        active = int(request.POST.get('active'))
        
        predictor = CardioRiskPredictor(model_file="modelTraining/model.pkl")
        percentage = predictor.predict(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        
        return render(request, 'results.html', {'percentage': percentage})
    else:
        return render(request, 'home.html')

def results(request):
    return render(request, 'results.html')
