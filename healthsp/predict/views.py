import os
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from django.shortcuts import render
from .utils import predict_disease, l1
from django.core.serializers.json import DjangoJSONEncoder
import json
from .utils import predict_disease, l1, symptom_data, get_prediction_stats
# Create your views here.

def home(request):
    return render(request,"home.html")

from django.core.serializers.json import DjangoJSONEncoder
import json
from .utils import predict_disease, l1, symptom_data, get_prediction_stats
def predict(request):
    context = {'symptoms_list': l1, 'symptom_descriptions': symptom_data}
    if 'prediction_history' not in request.session:
        request.session['prediction_history'] = []
    
    if request.method == 'POST':
        selected_symptoms = request.POST.getlist('symptoms[]')
        selected_symptoms = [s for s in selected_symptoms if s]
        if len(selected_symptoms) < 2:
            context['error'] = 'Please select at least 2 symptoms.'
        elif len(selected_symptoms) != len(set(selected_symptoms)):
            context['error'] = 'Duplicate symptoms are not allowed!'
        else:
            disease, department, precaution = predict_disease(selected_symptoms)
            request.session['prediction_history'].append(str(disease))
            request.session.modified = True
            stats = get_prediction_stats(request.session['prediction_history'])
            context.update({
                'disease': disease,
                'department': department,
                'precaution': precaution,
                'prediction_stats': json.dumps(stats, cls=DjangoJSONEncoder)
            })
        return render(request, 'prediction.html', context)
    
    stats = get_prediction_stats(request.session['prediction_history'])
    context['prediction_stats'] = json.dumps(stats, cls=DjangoJSONEncoder)
    return render(request, 'prediction.html', context)

def contact(request):
    return render(request,"Contact.html")


def about(request):
    return render(request,"About_us.html")


def article(request):
    return render(request,"article.html")
