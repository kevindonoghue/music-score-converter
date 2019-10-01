from data.generate_sample import generate_sample
import time

t = time.time()
sample_size = 50
multiplicity = 3

generate_sample(2*sample_size, multiplicity, 'halves_wholes_no_chords_no_rests_c', 224, 224, 'halves_wholes', 'halves_wholes', 'none', 'none', 0, key_number_choices=(0,))
generate_sample(2*sample_size, multiplicity, 'halves_wholes_chords_rests_c', 224, 224, 'halves_wholes', 'halves_wholes', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(2*sample_size, multiplicity, 'quarters_no_chords_no_rests_c', 224, 224, 'quarters', 'quarters', 'none', 'none', 0, key_number_choices=(0,))
generate_sample(2*sample_size, multiplicity, 'quarters_no_chords_rests_c', 224, 224, 'quarters', 'quarters', 'none', 'none', 0.2, key_number_choices=(0,))
generate_sample(3*sample_size, multiplicity, 'quarters_halves_chords_no_rests_c', 224, 224, 'quarters', 'halves_wholes', 'none', 'tight', 0, key_number_choices=(0,))
generate_sample(3*sample_size, multiplicity, 'halves_quarters_chords_no_rests_c', 224, 224, 'halves_wholes', 'quarters', 'tight', 'none', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_chords_no_rests_c', 224, 224, 'quarters', 'quarters', 'tight', 'tight', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_chords_rests_c', 224, 224, 'quarters', 'quarters', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_halves_chords_no_rests_c', 224, 224, 'quarters', 'halves_wholes', 'tight', 'tight', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'halves_quarters_chords_no_rests_c', 224, 224, 'halves_wholes', 'quarters', 'tight', 'tight', 0, key_number_choices=(0,))
generate_sample(2*sample_size, multiplicity, 'quarters_eight_halves_no_chords_rests_c', 224, 224, 'quarters_eighths', 'halves_wholes', 'none', 'none', 0.2, key_number_choices=(0,))
generate_sample(2*sample_size, multiplicity, 'quarters_eight_halves_chords_rests_c', 224, 224, 'quarters_eighths', 'halves_wholes', 'none', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'sixteenths_halves_no_chords_no_rests_c', 224, 224, 'sixteenths', 'halves_wholes', 'none', 'tight', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'halves_sixteenths_no_chords_no_rests_c', 224, 224, 'halves_wholes', 'sixteenths', 'tight', 'none', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_eighths_sixteenths_halves_no_chords_no_rests_c', 224, 224, 'quarters_eighths_sixteenths', 'halves_wholes', 'none', 'tight', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_eighths_sixteenths_halves_no_chords_rests_c', 224, 224, 'quarters_eighths_sixteenths', 'halves_wholes', 'none', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'halves_quarters_eighths_sixteenths_halves_no_chords_no_rests_c', 224, 224, 'halves_quarters_eighths_sixteenths', 'halves_wholes', 'none', 'tight', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'halves_quarters_eighths_sixteenths_halves_no_chords_rests_c', 224, 224, 'halves_quarters_eighths_sixteenths', 'halves_wholes', 'none', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'wholes_wholes_chords_rests_c', 224, 224, 'wholes', 'wholes', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_eighths_sixteenths_dots_no_chords_no_rests', 224, 224, 'quarters_eighths_sixteenths_dots', 'quarters_eighths_sixteenths_dots', 'none', 'none', 0, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_eighths_sixteenths_dots_no_chords_rests', 224, 224, 'quarters_eighths_sixteenths_dots', 'quarters_eighths_sixteenths_dots', 'none', 'none', 0.2, key_number_choices=(0,))
generate_sample(3*sample_size, multiplicity, 'quarters_eighths_sixteenths_dots_chords_rests', 224, 224, 'quarters_eighths_sixteenths_dots', 'quarters_eighths_sixteenths_dots', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_eighths_sixteenths_quarters_dots_chords_rests', 224, 224, 'quarters_eighths_sixteenths_dots', 'quarters', 'tight', 'tight', 0.2, key_number_choices=(0,))
generate_sample(sample_size, multiplicity, 'quarters_eighths_sixteenths_halves_dots_chords_rests', 224, 224, 'quarters_eighths_sixteenths_dots', 'halves_wholes', 'tight', 'tight', 0.2, key_number_choices=(0,))

with open('total_train_time.txt', 'w+') as f:
    f.write(str(time.time() - t))