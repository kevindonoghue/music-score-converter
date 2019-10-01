import numpy as np
# from .generate_rhythm import generate_rhythm
# from .generate_chords import generate_chords
from .generate_measure import generate_measure
from bs4 import BeautifulSoup


def generate_measure_for_score(measure_length, key_number, rest_prob, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key, measure_number):
    decorated_measure = generate_measure(measure_length, key_number, rest_prob, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key)
    measure = decorated_measure.find('measure')
    measure.attrs['number'] = str(measure_number)
    attributes = measure.find('attributes')
    attributes.extract()
    return measure


def generate_attributes(measure_length, key_number):
    # create the attributes tag to be placed in the first measure
    soup = BeautifulSoup('', 'xml')
    attributes = soup.new_tag('attributes')
    divisions = soup.new_tag('divisions')
    key = soup.new_tag('key')
    fifths = soup.new_tag('fifths')
    time = soup.new_tag('time')
    beats = soup.new_tag('beats')
    beat_type = soup.new_tag('beat-type')
    staves = soup.new_tag('staves')
    treble_clef = soup.new_tag('clef')
    bass_clef = soup.new_tag('clef')
    treble_sign = soup.new_tag('sign')
    treble_line = soup.new_tag('line')
    bass_sign = soup.new_tag('sign')
    bass_line = soup.new_tag('line')
    attributes.append(divisions)
    divisions.string = '4'
    attributes.append(key)
    key.append(fifths)
    fifths.string = str(key_number)
    attributes.append(time)
    time.append(beats)
    beats.string = str(int(measure_length/4))
    time.append(beat_type)
    beat_type.string = '4'
    attributes.append(staves)
    staves.string = '2'
    attributes.append(treble_clef)
    treble_clef['number']='1'
    treble_clef.append(treble_sign)
    treble_sign.string = 'G'
    treble_clef.append(treble_line)
    treble_line.string = '2'
    attributes.append(bass_clef)
    bass_clef['number'] = '2'
    bass_clef.append(bass_sign)
    bass_sign.string = 'F'
    bass_clef.append(bass_line)
    bass_line.string = '4'
    return attributes


def generate_score(num_measures, measure_length, key_number, rest_prob, treble_tp_key_choices=('complex',), bass_tp_key_choices=('complex',), treble_cp_key_choices=('complex',), bass_cp_key_choices=('complex',)):
    # generates a score num_measures measures long
    # measure_length is the number of sixteenth notes in a measure
    # see below for an example of chord_probs
    # the chord_probs[i] is the probability of adding a note off an i-7 interval (where the interval 1 is a unison)
    soup = BeautifulSoup('', 'xml')
    score_partwise = soup.new_tag('score-partwise', version='3.1')
    work = soup.new_tag('work')
    work_title = soup.new_tag('work-title')
    alpha = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890      ')
    text = list(np.random.choice(alpha, size=np.random.randint(8, 25)))
    text = ''.join(text)
    work_title.string = text
    score_partwise.append(work)
    work.append(work_title)
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

    attributes = generate_attributes(measure_length, key_number)

    for i in range(num_measures):
        n = np.random.choice(len(treble_tp_key_choices))
        treble_tp_key = treble_tp_key_choices[n]
        n = np.random.choice(len(bass_tp_key_choices))
        bass_tp_key = bass_tp_key_choices[n]
        n = np.random.choice(len(treble_cp_key_choices))
        treble_cp_key = treble_cp_key_choices[n]
        n = np.random.choice(len(bass_cp_key_choices))
        bass_cp_key = bass_cp_key_choices[n]
        measure = generate_measure_for_score(measure_length, key_number, rest_prob, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key, i+1)
        if i == 0:
            measure.insert(0, attributes)
        part.append(measure)

    return soup



chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
chord_probs = chord_probs / chord_probs.sum()
# print(generate_score(16, 4, 3, 0.2, chord_probs).prettify())

# with open('sample_score.musicxml', 'w+') as f:
#     f.write(str(generate_score(64, 16, 0, 0.2, treble_tp_key_choices=('quarters',))))