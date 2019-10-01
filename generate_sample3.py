from data.generate_sample import generate_sample


generate_sample(2500, 5, 'quarters_chords_no_rests_c', 224, 224, 'quarters', 'quarters', 'tight', 'tight', 0, key_number_choices=(0,))
generate_sample(2500, 5, 'quarters_chords_rests_c', 224, 224, 'quarters', 'quarters', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(2500, 5, 'quarters_halves_chords_no_rests_c', 224, 224, 'quarters', 'halves_wholes', 'tight', 'tight', 0, key_number_choices=(0,))
generate_sample(2500, 5, 'halves_quarters_chords_no_rests_c', 224, 224, 'halves_wholes', 'quarters', 'tight', 'tight', 0, key_number_choices=(0,))
