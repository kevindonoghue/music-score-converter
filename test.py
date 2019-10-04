import numpy as np
from data.generate_score import generate_score
from data.generate_bboxes import get_bboxes
from skimage import io, transform
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import subprocess
import os
import pathlib
from PIL import Image, ImageDraw
from data.generate_score_sample import generate_score_sample, tight_tp_choices, wide_tp_choices, cp_choices
    

generate_score_sample('asdf', 1, 64, ('complex',), ('complex',), ('complex',), ('complex',))