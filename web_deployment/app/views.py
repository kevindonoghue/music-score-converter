from django.shortcuts import render
import torch
from model import run_model
from model import Net
from .forms import UploadedPageForm
from django.http import HttpResponseRedirect
import json
import os
from handle_page import handle_page
from bs4 import BeautifulSoup, Tag
from django.conf import settings
import numpy as np

MEDIA_ROOT = settings.MEDIA_ROOT
BASE_DIR = settings.BASE_DIR

def get_blank_measure(measure_length):
    # need to set up a special case for the first measure
    blank_measure = Tag(name='measure')
    treble_note = Tag(name='note')
    treble_rest = Tag(name='rest')
    treble_duration = Tag(name='duration')
    treble_staff = Tag(name='staff')
    backup = Tag(name='backup')
    backup_duration = Tag(name='duration')
    bass_note = Tag(name='note')
    bass_rest = Tag(name='rest')
    bass_duration = Tag(name='duration')
    bass_staff = Tag(name='staff')
    treble_duration.string = str(measure_length)
    backup_duration.string = str(measure_length)
    bass_duration.string = str(measure_length)
    treble_staff.string = str(1)
    bass_staff.string = str(2)
    blank_measure.append(treble_note)
    treble_note.append(treble_rest)
    treble_note.append(treble_duration)
    treble_note.append(treble_staff)
    blank_measure.append(backup)
    blank_measure.append(backup_duration)
    blank_measure.append(bass_note)
    bass_note.append(bass_rest)
    bass_note.append(bass_duration)
    bass_note.append(bass_staff)
    return blank_measure
 
def create_musicxml(path, measure_length, key_number):
    """
    This function takes the path to an uploaded file, its measure_length, and its key number
    (usually info inputted by user) and passes the image through the first neural net to extract the measures,
    then passes each measure through the second neural net to convert it to xml.
    
    The handle_page function covers the first part and the run_model function covers the second.
    """
    handle_page(path, measure_length, key_number, os.path.join(MEDIA_ROOT, 'current_measures')) 
    measures = []
    
    # initialize the xml output
    soup = BeautifulSoup(features='xml')
    score_partwise = soup.new_tag('score-partwise', version='3.1')
    part_list = soup.new_tag('part-list')
    score_part = soup.new_tag('score-part', id='P1')
    part_name = soup.new_tag('part-name')
    soup.append(score_partwise)
    score_partwise.append(part_list)
    part_list.append(score_part)
    score_part.append(part_name)
    part_name.append('Piano')
    part = soup.new_tag('part', id='P1')
    score_partwise.append(part)

    # loop through each extracted measure and convert it to xml
    # if the conversion fails, return a blank measure
    for i in range(len(os.listdir(os.path.join(MEDIA_ROOT, 'current_measures')))):
        print('handling measure ', i+1)
        measure_soup = run_model(os.path.join(MEDIA_ROOT, 'current_measures', f'subimage{i}.png'), measure_length, key_number)
        if measure_soup:
            measure = measure_soup.find('measure')
            # only need the key and time sig info on the first measure
            if i != 0:
                attributes = measure.find('attributes')
                attributes.extract()
            measures.append(measure)
            print(f'measure {i+1} successful')
        else:
            blank_measure = get_blank_measure(measure_length)
            measures.append(blank_measure)
            print('error in measure ', i+1)
    for measure in measures:
        part.append(measure)

    # pick a random filename for the output
    filename = np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=16)
    filename = ''.join(filename)
    with open(os.path.join(MEDIA_ROOT, f'{filename}.musicxml'), 'w+') as f:
        f.write(str(soup))
    return filename
        

def example(request):
    """
    This view handles the curated example linked to on the frontpage.
    """
    path = os.path.join(MEDIA_ROOT, 'minuet_larger.png')
    key_number = 0
    measure_length = 12
    output_filename = create_musicxml(path, measure_length, key_number)
    return HttpResponseRedirect(f'/media/{output_filename}.musicxml')
    

def homepage(request):
    """
    This view handles the main page of the website.
    For a post request, it calls the create_musicxml function then redirects to the generated xml file.
    For a get request, it displays the form for uploading a file.
    """
    if request.method == 'POST':
        form = UploadedPageForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            path = os.path.join(MEDIA_ROOT, file.page.name)
            # key_number = int(str(file.key))
            key_number = 0
            time_sig_dict = {'2/4': 8, '3/4': 12, '4/4': 16}
            measure_length = time_sig_dict[file.time_signature]
            output_filename = create_musicxml(path, measure_length, key_number)
            return HttpResponseRedirect(f'media/{output_filename}.musicxml')
    elif request.method == 'GET':
        form = UploadedPageForm()
    context = {'form': form}
    return render(request, 'homepage.html', context)