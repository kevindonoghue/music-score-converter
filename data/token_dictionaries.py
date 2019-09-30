tokens = [
    '<START>', '<END>', '<PAD>', 'measure', 'note', 'pitch', 'step', 'alter',
    'octave', 'duration', 'type', 'rest', 'dot', 'staff', 'notations', 'slur',
    'ff', 'f', 'mf', 'mp', 'p', 'pp', 'backup', 'chord'] + list('ABCDEFG') + ['-1'] + list('0123456789') + ['10', '11', '12', '13', '14', '15', '16']

word_to_ix = {word: i for i, word in enumerate(tokens)}
ix_to_word = {str(i): word for i, word in enumerate(tokens)}