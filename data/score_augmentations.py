import numpy as np
from PIL import Image
from scipy.ndimage.filters import correlate
from skimage import transform


# random_augmentation takes a numpy array of dimension 2 with entries between 0 and 1, distorts it, and rescales it
# the rest of the functions are helper functions

# bbox_arr is a an array of shape (N, 2, 2) where there are N bboxes, and bbox_arr[i, :, :] is [[x1, y1], [x2, y2]]



def rgb_to_grayscale(x):
    # x is an numpy array with shape (a, b, 3)
    a, b, _ = x.shape
    weights = np.array([[0.2125, 0.7154, 0.0721]]).T
    return (x @ weights).reshape(a, b)


########################### augmentation functions ###########################
def clip_left(x, bbox_arr, p):
    k = int(x.shape[1]*p)
    x = x[:, k:]
    bbox_arr[:, :, 0] = (bbox_arr[:, :, 0] - p)/(1 - p)
    return x, bbox_arr

def clip_right(x, bbox_arr, p):
    k = int(x.shape[1]*p)
    x = x[:, :x.shape[1]-k]
    bbox_arr[:, :, 0] = bbox_arr[:, :, 0]/(1-p)
    return x, bbox_arr

def extend_left(x, bbox_arr, p):
    k = int(x.shape[1]*p)
    x = np.concatenate([np.ones((x.shape[0], k)), x], axis=1)
    bbox_arr[:, :, 0] = (bbox_arr[:, :, 0] + p)/(1 + p)
    return x, bbox_arr

def extend_right(x, bbox_arr, p):
    k = int(x.shape[1]*p)
    x = np.concatenate([x, np.ones((x.shape[0], k))], axis=1)
    bbox_arr[:, :, 0] = bbox_arr[:, :, 0]/(1+p)
    return x, bbox_arr

def clip_up(x, bbox_arr, p):
    k = int(x.shape[0]*p)
    x = x[k:, :]
    bbox_arr[:, :, 1] = (bbox_arr[:, :, 1] - p)/(1 - p)
    return x, bbox_arr

def clip_down(x, bbox_arr, p):
    k = int(x.shape[0]*p)
    x = x[:x.shape[0]-k, :]
    bbox_arr[:, :, 1] = bbox_arr[:, :, 1]/(1-p)
    return x, bbox_arr

def extend_up(x, bbox_arr, p):
    k = int(x.shape[0]*p)
    x = np.concatenate([np.ones((k, x.shape[1])), x])
    bbox_arr[:, :, 1] = (bbox_arr[:, :, 1] + p)/(1 + p)
    return x, bbox_arr


def extend_down(x, bbox_arr, p):
    k = int(x.shape[0]*p)
    x = np.concatenate([x, np.ones((k, x.shape[1]))])
    bbox_arr[:, :, 1] = bbox_arr[:, :, 1]/(1 + p)
    return x, bbox_arr


def horizontal_jiggle(x, N, p):
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
    if color == 'black':
        noise = np.random.choice([0, 1], size=x.shape, p=[p, 1-p])
        return np.minimum(x, noise)
    else:
        noise = np.random.choice([0, 1], size=x.shape, p=[1-p, p])
        return np.maximum(x, noise)

def make_bw(x):
    return np.around(x)

def convert_to_1(x):
    image = Image.fromarray(x*255)
    image = image.convert('1')
    x = np.array(image)/255
    return x

def thicken(x, dims):
    x = 1-x
    k = np.ones(dims)
    x = np.minimum(correlate(x, k), 1)
    return 1-x

def add_splotch(x, N, r):
    a = np.random.randint(x.shape[0])
    b = np.random.randint(x.shape[1])
    splotch_points = np.around(np.array([a, b]) + np.random.randn(N, 2)*r)
    splotch_points = splotch_points.astype(int)
    for point in splotch_points:
        if 0 <= point[0] < x.shape[0] and 0 <= point[1] < x.shape[1]:
            x[point[0], point[1]] = np.minimum(x[point[0], point[1]], 0)
    return x


########################### random augmentation functions ###########################   
def random_resize(x, bbox_arr):
    left = np.random.uniform(low=-0.1, high=0.1)
    right = np.random.uniform(low=-0.1, high=0.1)
    up = np.random.uniform(low=-0.1, high=0.1)
    down = np.random.uniform(low=-0.1, high=0.1)
    
    if left > 0:
        x, bbox_arr = extend_left(x, bbox_arr, left)
    else:
        x, bbox_arr = clip_left(x, bbox_arr, np.abs(left))
    if right > 0:
        x, bbox_arr = extend_right(x, bbox_arr, right)
    else:
        x, bbox_arr = clip_right(x, bbox_arr, np.abs(right))
    if up > 0:
        x, bbox_arr = extend_up(x, bbox_arr, up)
    else:
        x, bbox_arr = clip_up(x, bbox_arr, np.abs(up))
    if down > 0:
        x, bbox_arr = extend_down(x, bbox_arr, down)
    else:
        x, bbox_arr = clip_down(x, bbox_arr, np.abs(down))
    return x, bbox_arr

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
    dim_choices = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
    dims = dim_choices[np.random.randint(len(dim_choices))]
    x = thicken(x, dims)
    return x

########################### put the random augmentation functions together ########################### 
def random_pre_augmentation(x, bbox_arr):
    if np.random.rand() < 0.9:
        x, bbox_arr = random_resize(x, bbox_arr)
    if np.random.rand() < 0.6:
        x = random_thicken(x)
    if np.random.rand() < 0.3:
        x = random_splotches(x)
    if np.random.rand() < 0.6:
        x = random_noise(x)
    if np.random.rand() < 0.3:
        x = random_jiggle(x)
    return x, bbox_arr

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

def random_score_augmentation(x, bbox_arr, height, width):
    x, bbox_arr = random_pre_augmentation(x, bbox_arr)
    x = rescale(x, height, width)
    x = random_post_augmentation(x)
    return x, bbox_arr