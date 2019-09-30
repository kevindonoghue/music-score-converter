import numpy as np
from .generate_rhythm import generate_rhythm
from .generate_chords import generate_chords
from bs4 import BeautifulSoup
import subprocess
import pathlib
import os
import time



def generate_measure(measure_length, key_number, rest_prob, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key):
    # measure_length is the length of the measure in 16th notes
    # clef is either 'treble' or 'bass'
    # key_number is an integer in the range(-7, 8)
    # tp_key and cp_key are keys for rhythm and chord probabilities (see generate_chords.py and generate_rhythm.py)
    tp_key = dict()
    tp_key['treble'] = treble_tp_key
    tp_key['bass'] = bass_tp_key
    
    cp_key = dict()
    cp_key['treble'] = treble_cp_key
    cp_key['bass'] = bass_cp_key
    
    rhythm = dict()
    chords = dict()
    for clef_type in ('treble', 'bass'):
        rhythm[clef_type] = generate_rhythm(tp_key[clef_type], measure_length)
        chords[clef_type] = generate_chords(len(rhythm[clef_type]), clef_type, cp_key[clef_type])
        for i, r in enumerate(rhythm[clef_type]):
            if r in (1, 2, 4, 8, 16):
                rest_bool = np.random.choice([True, False], p=[rest_prob, 1-rest_prob])
                if rest_bool:
                    chords[clef_type][i] = 'rest'
    
    soup = BeautifulSoup('', 'xml')
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
    measure = soup.new_tag('measure', number='1')
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


    score_partwise.append(part)
    part.append(measure)
    measure.append(attributes)
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

    def note_to_soup(s, dur, rest, staff_number):
        # here s is a string like 'E5'
        # dot and rest are bools
        note = soup.new_tag('note')
        if rest:
            pitch = None
            step = None
            alter = None
            octave = None
            rest = soup.new_tag('rest')
        else:
            pitch = soup.new_tag('pitch')
            step = soup.new_tag('step')
            step.string = s[0]
            # alt = str(np.random.choice(['-1', '0', '1'], p=[0.15, 0.7, 0.15]))
            alt = str(np.random.choice(['-1', '0', '1']))
            if alt != '0':
                alter = soup.new_tag('alter')
                alter.string = str(alt)
            else:
                alter = None
            octave = soup.new_tag('octave')
            octave.string = s[1]
            rest = None

        if dur == 1:
            type_string = '16th'
            dot = None
        elif dur == 2:
            type_string = 'eighth'
            dot = None
        elif dur == 3:
            type_string = 'eighth'
            dot = soup.new_tag('dot')
        elif dur == 4:
            type_string = 'quarter'
            dot = None
        elif dur == 6:
            type_string = 'quarter'
            dot = soup.new_tag('dot')
        elif dur == 8:
            type_string = 'half'
            dot = None
        elif dur == 12:
            type_string = 'half'
            dot = soup.new_tag('dot')
        elif dur == 16:
            type_string = 'whole'
            dot = None

        duration = soup.new_tag('duration')
        duration.string = str(dur)

        type_ = soup.new_tag('type')
        type_.string = type_string

        if pitch:
            note.append(pitch)
            pitch.append(step)
            if alter:
                pitch.append(alter)
            pitch.append(octave)
            
        if rest:
            note.append(rest)
        note.append(duration)
        note.append(type_)
        if dot: # should always be false for rests
            note.append(dot)
        staff = soup.new_tag('staff')
        note.append(staff)
        staff.string = str(staff_number)
        return note

    def chord_to_soup(x, dur, staff_number):
        # here x is either an array of pitches like ['E5', 'A6'] or the string 'rest'
        # returns a list of note tags
        if x == 'rest':
            return [note_to_soup(None, dur, True, staff_number)]
        else:
            notes = [note_to_soup(s, dur, False, staff_number) for s in x]
            if len(notes) > 1:
                for note in notes[1:]:
                    note.insert(0, soup.new_tag('chord'))
            return notes

    for clef_type in ('treble', 'bass'):
        staff_number = 1 if clef_type == 'treble' else 2
        chords[clef_type] = [chord_to_soup(chord, rhythm[clef_type][i], staff_number) for i, chord in enumerate(chords[clef_type])]
        if np.random.rand() < 0 and len(chords[clef_type]) > 1:
            slur_indices = np.random.choice(len(chords[clef_type]), size=2, replace=False)
            initial_index = np.min(slur_indices)
            final_index = np.max(slur_indices)
            initial_note = chords[clef_type][initial_index][0]
            final_note = chords[clef_type][final_index][0]
            if not initial_note.find_all('rest') and not final_note.find_all('rest'): 
                notations = soup.new_tag('notations')
                slur = soup.new_tag('slur')
                slur['type'] = 'start'
                initial_note.append(notations)
                notations.append(slur)
                notations = soup.new_tag('notations')
                slur = soup.new_tag('slur')
                slur['type'] = 'stop'
                final_note.append(notations)
                notations.append(slur)
        for i, chord in enumerate(chords[clef_type]):
            if np.random.rand() < 0.3 and staff_number == 1:
                direction = soup.new_tag('direction')
                direction['placement'] = 'below'
                direction_type = soup.new_tag('direction-type')
                dynamics = soup.new_tag('dynamics')
                dynamic_tag_name = np.random.choice(['ff', 'f', 'mf', 'mp', 'p', 'pp'], p=[0.05, 0.3, 0.15, 0.15, 0.3, 0.05 ])
                dynamic_tag = soup.new_tag(dynamic_tag_name)
                measure.append(direction)
                direction.append(direction_type)
                direction_type.append(dynamics)
                dynamics.append(dynamic_tag)
                staff = soup.new_tag('staff')
                staff.string = '1'
            for note in chord:
                measure.append(note)
        if staff_number == 1:
            backup = soup.new_tag('backup')
            duration = soup.new_tag('duration')
            duration.string = str(measure_length)
            backup.append(duration)
            measure.append(backup)

    return soup
    
    
    
        
# if not os.path.exists('measures/'):
#     os.mkdir('measures/') 
# for tp_key in tp:
#     for cp_key in chord_probs:
#         prefix = f'measures/sample_measure_{tp_key}_{cp_key}'
#         t = time.time()
#         with open(f'{prefix}.musicxml', 'w+') as f:
#             measure_length = np.random.choice([8, 12, 16])
#             key_number = 0
#             rest_prob = 0
#             soup = generate_measure(measure_length, key_number, rest_prob, tp_key, tp_key, cp_key, cp_key)
#             f.write(str(soup))
#         subprocess.call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', f'{prefix}.musicxml', '-o', f'{prefix}.mscz'])
#         subprocess.call(['mscore', f'{prefix}.mscz', '-o', f'{prefix}.png'])
#         subprocess.call(['mscore', f'{prefix}.mscz', '-o', f'{prefix}.svg'])
#         print(time.time() - t)
