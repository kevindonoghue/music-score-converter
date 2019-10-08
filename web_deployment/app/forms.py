from django import forms
from .models import UploadedPage


class UploadedPageForm(forms.ModelForm):
    key = forms.CharField(max_length=1, required=False)
    time_signature = forms.CharField(max_length=3, required=True, widget=forms.TextInput(attrs={'size': 3}))
    
    class Meta:
        model = UploadedPage
        fields = ['page', 'key', 'time_signature']