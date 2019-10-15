import numpy as np

# initialize lists pitches for the treble and bass clefts
notes = dict()
notes['treble'] = ['G3', 'A3', 'B3'] + [f'{ch}4' for ch in list('CDEFGAB')] + [f'{ch}5' for ch in list('CDEFGAB')] + ['C6', 'D6']
notes['bass'] = [f'{ch}2' for ch in list('CDEFGAB')] + [f'{ch}3' for ch in list('CDEFGAB')] + [f'{ch}4' for ch in list('CDEF')]

# cp stands for chord probs
# these are used to in generate_chords to determine how many notes/how wide intervals the chord should have
cp = dict()
cp['complex'] = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
cp['tight'] = np.array([0, 0, 0, 20, 20, 20, 0, 100, 0, 20, 20, 20, 0, 0, 0])
cp['dense'] = np.array([30, 0, 30, 30, 30, 30, 0, 100, 0, 30, 30, 30, 30, 0, 30])
cp['none'] = np.array([0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0])
for key in cp:
    cp[key] = cp[key]/cp[key].sum()


# determine probabilities for the bottom note of each chord
start_probs = dict()
start_probs['treble'] = dict()
start_probs['bass'] = dict()

for i, x in enumerate(notes['bass']):
    if i < notes['bass'].index('E2'):
        start_probs['bass'][x] = 1
    elif i > notes['bass'].index('C4'):
        start_probs['bass'][x] = 1
    else:
        start_probs['bass'][x] = 10

bass_total = np.sum([x for _, x in start_probs['bass'].items()])
for note in start_probs['bass']:
    start_probs['bass'][note] /= bass_total

for i, x in enumerate(notes['treble']):
    if i < notes['treble'].index('C4'):
        start_probs['treble'][x] = 1
    elif i > notes['treble'].index('A5'):
        start_probs['treble'][x] = 1
    else:
        start_probs['treble'][x] = 10

treble_total = np.sum([x for _, x in start_probs['treble'].items()])
for note in start_probs['treble']:
    start_probs['treble'][note] /= treble_total

# determine probabilities for how the notes in the chord change from one chord to the next
# to be realistic you don't want the notes to jump around too much
# ptp stands for pitch transition probabilities:
ptp = dict()
for i in range(-2, 3):
    ptp[i] = 75
for i in list(range(-4, -2)) + list(range(3, 5)):
    ptp[i] = 20
for i in list(range(-8, -4)) + list(range(5, 9)):
    ptp[i] = 4
for i in list(range(-10, -8)) + list(range(9, 11)):
    ptp[i] = 1

ptp_total = np.sum([x for _, x in ptp.items()])
for interval in ptp:
    ptp[interval] /= ptp_total


# for later use, record the items of these dictionaries
start_probs_items = dict()
start_probs_items['treble'] = start_probs['treble'].items()
start_probs_items['bass'] = start_probs['bass'].items()
ptp_items = ptp.items()



def generate_chords(num_chords, clef, chord_prob_key):
    # clef is either 'treble' or 'bass'
    # chord_prob_key is key for the dictionary cp defined at the top of this file
    # outputs a list of length num_chords, each of which is a list like ['F5'] or ['A5', 'C6']
    # each list in the output should be thought of as the notes in a chord
    # always returns the notes from lowest to highest
    chord_probs = cp[chord_prob_key]
    
    pitches = [x for x, _ in start_probs_items[clef]]
    start_probs = [y for _, y in start_probs_items[clef]]
    start_pitch = np.random.choice(pitches, p=start_probs)
    intervals = [x for x, _ in ptp_items]
    interval_probs = [y for _, y in ptp_items]

    generated_pitches = [[start_pitch]]
    while len(generated_pitches) < num_chords:
        reference_pitch = generated_pitches[-1][-1]

        interval = np.random.choice(intervals, p=interval_probs)
        last_pitch_ix = pitches.index(reference_pitch)
        if last_pitch_ix + interval in range(len(pitches)):
            generated_pitches.append([pitches[last_pitch_ix + interval]])

    for i, x in enumerate(generated_pitches):
        ix = pitches.index(x[0])
        for j, p in enumerate(chord_probs):
            if np.random.rand() < p and j != 7:
                interval = -7 + j
                if ix + interval in range(len(pitches)):
                    generated_pitches[i].append(pitches[ix + interval])

    for chord in generated_pitches:
        chord.sort(key=lambda x: notes[clef].index(x))
    return generated_pitches




# chords = generate_chords(20, 'treble', 'tight')
# for chord in chords:
#     print(chord)