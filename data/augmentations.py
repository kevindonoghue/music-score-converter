import numpy as np
from PIL import Image
from scipy.ndimage.filters import correlate
from skimage import transform


"""
random_augmentation takes a numpy array of dimension 2 with entries between 0 and 1, distorts it, and rescales it.
The rest of the functions are helper functions.
"""

########################### augmentation functions ###########################
def clip_left(x, k):
    """removes k pixels from the left of the image"""
    return x[:, k:]

def clip_right(x, k):
    """removes k pixels from the right of the image"""
    return x[:, :x.shape[1]-k]

def extend_left(x, k):
    """adds k pixels to the left of the image"""
    left_col = x[:, 0].reshape(-1, 1)
    tiled = np.tile(left_col, (1, k))
    return np.concatenate([tiled, x], axis=1)

def extend_right(x, k):
    """adds k pixels to the right of the image"""
    right_col = x[:, 0].reshape(-1, 1)
    tiled = np.tile(right_col, (1, k))
    return np.concatenate([x, tiled], axis=1)

def extend_up(x, k):
    """adds k pixels to the top of the image"""
    return np.concatenate([np.ones((k, x.shape[1])), x])

def extend_down(x, k):
    """adds k pixels to the bottom of the image"""
    return np.concatenate([x, np.ones((k, x.shape[1]))])

def horizontal_jiggle(x, N, p):
    """adds some distortion to the image"""
    jiggle_direction = np.random.choice([1, -1])
    y = np.random.uniform(size=(N, x.shape[0]))
    for i in range(N):
        mask = y[i] < p
        if jiggle_direction == 1:
            x[mask, 1:] = x[mask, :-1]
        else:
            x[mask, :-1] = x[mask, 1:]
    return x

def vertical_jiggle(x, N, p):
    """adds some distortion to the image"""
    jiggle_direction = np.random.choice([1, -1])
    y = np.random.uniform(size=(N, x.shape[1]))
    for i in range(N):
        mask = y[i] < p
        if jiggle_direction == 1:
            x[1:, mask] = x[:-1, mask]
        else:
            x[:-1, mask] = x[1:, mask]
    return x

def add_noise(x, p, color='black'):
    """adds noise to the image"""
    if color == 'black':
        noise = np.random.choice([0, 1], size=x.shape, p=[p, 1-p])
        return np.minimum(x, noise)
    else:
        noise = np.random.choice([0, 1], size=x.shape, p=[1-p, p])
        return np.maximum(x, noise)

def make_bw(x):
    """changes the image from grayscale to black and white"""
    return np.around(x)

def thicken(x, dims):
    """Thickens the lines in the image with a correlation filter of 1s. dims is the dimensions of the filter"""
    x = 1-x
    k = np.ones(dims)
    x = np.minimum(correlate(x, k), 1)
    return 1-x

def add_splotch(x, N, r):
    """add something like an inkblot to the page"""
    a = np.random.randint(x.shape[0])
    b = np.random.randint(x.shape[1])
    splotch_points = np.around(np.array([a, b]) + np.random.randn(N, 2)*r)
    splotch_points = splotch_points.astype(int)
    for point in splotch_points:
        if 0 <= point[0] < x.shape[0] and 0 <= point[1] < x.shape[1]:
            x[point[0], point[1]] = np.minimum(x[point[0], point[1]], 0)
    return x
            
def shrink_whitespace(x, p):
    """Randomly decrease the vertical space between the two staves in the image"""
    white_rows = np.nonzero(x.mean(axis=1) == 1)[0]
    white_rows = white_rows[(white_rows >= int(x.shape[0]*3/8)) & (white_rows < int(x.shape[0]*5/8))]
    excluded_rows = np.random.choice(white_rows, size=int(p*len(white_rows)), replace=False)
    mask = np.ones(x.shape[0])
    mask[excluded_rows] -= 1
    included_rows = np.nonzero(mask)[0]
    return x[included_rows]

def shrink_note_spacing(x, p):
    """Randomly decrase the horizontal space between the notes in the image"""
    empty_columns = np.nonzero((1-x).sum(axis=0) <= 12)[0]
    excluded_columns = np.random.choice(empty_columns, size=int(p*len(empty_columns)), replace=False)
    mask = np.ones(x.shape[1])
    mask[excluded_columns] -= 1
    included_columns = np.nonzero(mask)[0]
    return x[:, included_columns]



########################### random augmentation functions ###########################
"""These functions apply previous functions with random parameters"""
def random_shrink_whitespace(x):
    p = np.random.rand()
    x = shrink_whitespace(x, p)
    return x
    
def random_shrink_note_spacing(x):
    p = np.minimum(np.abs(np.random.randn()*0.3), 1)
    x = shrink_note_spacing(x, p)
    return x
    
def random_resize(x):
    left = int(np.maximum(-1, x.shape[1]*np.random.randn()*0.1))
    right = int(np.maximum(-10, x.shape[1]*np.random.randn()*0.1))
    vertical_base = np.abs(x.shape[0]*np.random.randn()*0.3)
    top = int(np.maximum(vertical_base + np.random.randn()*0.1, 0))
    bottom = int(np.maximum(vertical_base + np.random.randn()*0.15, 0))
    # top = int(np.abs(x.shape[0]*np.random.randn()*0.3))
    # bottom = int(np.abs(x.shape[0]*np.random.randn()*0.3))
    if left > 0:
        x = extend_left(x, left)
    else:
        x = clip_left(x, np.abs(left))
    if right > 0:
        x = extend_right(x, right)
    else:
        x = clip_right(x, np.abs(right))
    x = extend_up(x, top)
    x = extend_down(x, bottom)
    return x

def random_splotches(x):
    n = np.random.randint(5)
    for _ in range(n):
        N = np.random.randint(200, 300)
        r = np.random.uniform(0.5, 1)
        x = add_splotch(x, N, r)
    return x

def random_noise(x):
    color = np.random.choice(['black', 'white'])
    if color == 'black':
        p = np.abs(np.random.randn()*0.01)
#         p = np.random.uniform(0, 0.01)
    else:
        p = np.random.uniform(0, 0.05)
    x = add_noise(x, p, color)
    return x

def random_jiggle(x):
    N = np.random.randint(0, 2)
    p = np.random.randn()*0.05
    x = vertical_jiggle(x, N, p)
    N = np.random.randint(0, 2)
    p = np.abs(np.random.randn()*0.2)
    x = horizontal_jiggle(x, N, p)
    return x

def random_thicken(x):
    dim_choices = [(2, 1), (1, 2), (2, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    dims = dim_choices[np.random.randint(len(dim_choices))]
    x = thicken(x, dims)
    return x

########################### put the random augmentation functions together ########################### 
def random_pre_augmentation(x):
    if np.random.rand() < 0.8:
        x = random_shrink_whitespace(x)
    if np.random.rand() < 0.8:
        x = random_shrink_note_spacing(x)
    if np.random.rand() < 0.9:
        x = random_resize(x)
    if np.random.rand() < 0.5:
        x = random_thicken(x)
    if np.random.rand() < 0.3:
        x = random_splotches(x)
    if np.random.rand() < 0.6:
        x = random_noise(x)
    if np.random.rand() < 0.3:
        x = random_jiggle(x)
    if np.random.rand() < 0.3:
        x = transform.rotate(x, np.random.randn()*0.6, cval=1)
    return x

def rescale(x, height, width):
    x = transform.resize(x, (height, width))
    return x

def random_post_augmentation(x):
    if np.random.rand() < 0.5:
        x = random_noise(x)
    if np.random.rand() < 0.5:
        x = random_jiggle(x)
    if np.random.rand() < 0.3:
        x = random_splotches(x)
    return x

def random_augmentation(x, height, width):
    """This function randomly distorts the image, and resizes it to shape (height, width).
    x should have dimension 2 and be made entries between 0 and 1.
    """
    x = random_pre_augmentation(x)
    x = rescale(x, height, width)
    x = random_post_augmentation(x)
    return x