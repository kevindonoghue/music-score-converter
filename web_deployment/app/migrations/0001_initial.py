# Generated by Django 2.2.5 on 2019-10-06 21:29

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedMeasure',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('measure', models.FileField(upload_to='', verbose_name='Measure')),
                ('key', models.CharField(max_length=64)),
                ('time_signature', models.CharField(max_length=64)),
            ],
        ),
        migrations.CreateModel(
            name='UploadedPage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('page', models.FileField(upload_to='', verbose_name='Page')),
                ('key', models.CharField(max_length=64)),
                ('time_signature', models.CharField(max_length=64)),
            ],
        ),
    ]