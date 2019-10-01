from data.generate_sample import generate_sample


generate_sample(2500, 5, 'halves_wholes_quarters_chords_rests_c', 224, 224, 'halves_wholes', 'quarters', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(2500, 5, 'halves_wholes_quarters_complex_chords_c', 224, 224, 'halves_wholes', 'quarters', 'complex', 'complex', 0, key_number_choices=(0,))
generate_sample(2500, 5, 'quarters_no_chords_no_rests_c', 224, 224, 'quarters', 'quarters', 'none', 'none', 0, key_number_choices=(0,))
generate_sample(2500, 5, 'halves_quarters_eighths_sixteenths_chords_rests_c', 224, 224, 'quarters_eights', 'halves_quarters_eights_sixteenths', 'none', 'tight', 0.2, key_number_choices=(0,))