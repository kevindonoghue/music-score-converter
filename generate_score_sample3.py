from data.generate_score_sample import generate_score_sample, tight_tp_choices, wide_tp_choices, cp_choices
    

total = 1000
generate_score_sample('scores11', int(total/4), 64, ('halves_wholes',), ('quarters',), ('complex',), ('complex',))
generate_score_sample('scores12', int(total/4), 64, ('eighths',), ('eighths',), ('none',), ('none',))
generate_score_sample('scores13', int(total/4), 64, wide_tp_choices, wide_tp_choices, ('none',), ('none',))
generate_score_sample('scores14', int(total/4), 64, wide_tp_choices, wide_tp_choices, ('complex',), ('complex',))
