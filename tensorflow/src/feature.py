import numpy as np
from tqdm.auto import tqdm
def create_feature_statistics(X):
    
    # landmark indices in original data
    LIPS_IDXS0 = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])
    LEFT_HAND_IDXS0  = np.arange(468,489)
    RIGHT_HAND_IDXS0 = np.arange(522,543)
    POSE_IDXS0       = np.arange(502, 512)
    LANDMARK_IDXS0   = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0))
    HAND_IDXS0       = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
    N_COLS           = LANDMARK_IDXS0.size
    
    # Landmark indices in processed data
    LIPS_IDXS       = np.argwhere(np.isin(LANDMARK_IDXS0, LIPS_IDXS0)).squeeze() 
    LEFT_HAND_IDXS  = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
    RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
    HAND_IDXS       = np.argwhere(np.isin(LANDMARK_IDXS0, HAND_IDXS0)).squeeze()
    POSE_IDXS       = np.argwhere(np.isin(LANDMARK_IDXS0, POSE_IDXS0)).squeeze()
    
    # LIPS
    LIPS_MEAN_X  = np.zeros([LIPS_IDXS.size], dtype=np.float32)
    LIPS_MEAN_Y  = np.zeros([LIPS_IDXS.size], dtype=np.float32)
    LIPS_STD_X   = np.zeros([LIPS_IDXS.size], dtype=np.float32)
    LIPS_STD_Y   = np.zeros([LIPS_IDXS.size], dtype=np.float32)

    for col, ll in enumerate(tqdm( np.transpose(X[:,:,LIPS_IDXS], [2,3,0,1]).reshape([LIPS_IDXS.size, 3, -1]) )):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0: # X
                LIPS_MEAN_X[col] = v.mean()
                LIPS_STD_X[col] = v.std()
            if dim == 1: # Y
                LIPS_MEAN_Y[col] = v.mean()
                LIPS_STD_Y[col] = v.std()
            
    LIPS_MEAN = np.array([LIPS_MEAN_X, LIPS_MEAN_Y]).T
    LIPS_STD = np.array([LIPS_STD_X, LIPS_STD_Y]).T
    
    # LEFT HAND
    LEFT_HANDS_MEAN_X = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
    LEFT_HANDS_MEAN_Y = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
    LEFT_HANDS_STD_X = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
    LEFT_HANDS_STD_Y = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
    # RIGHT HAND
    RIGHT_HANDS_MEAN_X = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)
    RIGHT_HANDS_MEAN_Y = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)
    RIGHT_HANDS_STD_X = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)
    RIGHT_HANDS_STD_Y = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)

    for col, ll in enumerate(tqdm( np.transpose(X[:,:,HAND_IDXS], [2,3,0,1]).reshape([HAND_IDXS.size, 3, -1]) )):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0: # X
                if col < RIGHT_HAND_IDXS.size: # LEFT HAND
                    LEFT_HANDS_MEAN_X[col] = v.mean()
                    LEFT_HANDS_STD_X[col] = v.std()
                else:
                    RIGHT_HANDS_MEAN_X[col - LEFT_HAND_IDXS.size] = v.mean()
                    RIGHT_HANDS_STD_X[col - LEFT_HAND_IDXS.size] = v.std()
            if dim == 1: # Y
                if col < RIGHT_HAND_IDXS.size: # LEFT HAND
                    LEFT_HANDS_MEAN_Y[col] = v.mean()
                    LEFT_HANDS_STD_Y[col] = v.std()
                else: # RIGHT HAND
                    RIGHT_HANDS_MEAN_Y[col - LEFT_HAND_IDXS.size] = v.mean()
                    RIGHT_HANDS_STD_Y[col - LEFT_HAND_IDXS.size] = v.std()
            
    LEFT_HANDS_MEAN = np.array([LEFT_HANDS_MEAN_X, LEFT_HANDS_MEAN_Y]).T
    LEFT_HANDS_STD = np.array([LEFT_HANDS_STD_X, LEFT_HANDS_STD_Y]).T
    RIGHT_HANDS_MEAN = np.array([RIGHT_HANDS_MEAN_X, RIGHT_HANDS_MEAN_Y]).T
    RIGHT_HANDS_STD = np.array([RIGHT_HANDS_STD_X, RIGHT_HANDS_STD_Y]).T

    # POSE
    POSE_MEAN_X = np.zeros([POSE_IDXS.size], dtype=np.float32)
    POSE_MEAN_Y = np.zeros([POSE_IDXS.size], dtype=np.float32)
    POSE_STD_X = np.zeros([POSE_IDXS.size], dtype=np.float32)
    POSE_STD_Y = np.zeros([POSE_IDXS.size], dtype=np.float32)

    for col, ll in enumerate(tqdm( np.transpose(X[:,:,POSE_IDXS], [2,3,0,1]).reshape([POSE_IDXS.size, 3, -1]) )):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0: # X
                POSE_MEAN_X[col] = v.mean()
                POSE_STD_X[col] = v.std()
            if dim == 1: # Y
                POSE_MEAN_Y[col] = v.mean()
                POSE_STD_Y[col] = v.std()
            
    POSE_MEAN = np.array([POSE_MEAN_X, POSE_MEAN_Y]).T
    POSE_STD = np.array([POSE_STD_X, POSE_STD_Y]).T
    
    LIPS_START = 0
    LEFT_HAND_START = LIPS_IDXS.size
    RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
    POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size
    
    statistics = {"lips": (LIPS_START, LIPS_MEAN, LIPS_STD),
                  "left_hand": (LEFT_HAND_START, LEFT_HANDS_MEAN, LEFT_HANDS_STD),
                  "right_hand": (RIGHT_HAND_START, RIGHT_HANDS_MEAN, RIGHT_HANDS_STD),
                  "pose": (POSE_START, POSE_MEAN, POSE_STD)}
    
    return statistics



        