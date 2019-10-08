from django.db import models
from django.core.exceptions import ValidationError

def time_sig_validator(value):
    if value not in ('2/4', '3/4', '4/4'):
        raise ValidationError('Enter 2/4, 3/4, or 4/4')


class UploadedPage(models.Model):
    page = models.FileField('Page')
    key = models.CharField(max_length=1)
    time_signature = models.CharField(max_length=3, validators=[time_sig_validator])