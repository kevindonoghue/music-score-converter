from data.generate_score_sample import generate_score_sample, tight_tp_choices, wide_tp_choices, cp_choices
    

total = 5000
generate_score_sample('scores_tight', int(total/2), 64, wide_tp_choices, tight_tp_choices, cp_choices, cp_choices)
generate_score_sample('scores_tight', int(total/2), 64, tight_tp_choices, wide_tp_choices, cp_choices, cp_choices)