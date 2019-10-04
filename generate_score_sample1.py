from data.generate_score_sample import generate_score_sample, tight_tp_choices, wide_tp_choices, cp_choices
    

total = 1000
generate_score_sample('scores2', int(total/2), 64, tight_tp_choices, tight_tp_choices, cp_choices, cp_choices)
generate_score_sample('scores3', int(total/4), 64, ('quarters'), ('quarters',), ('none',), ('none',))
generate_score_sample('scores8', int(total/4), 64, ('halves_wholes'), ('quarters'), ('none',), ('none',))