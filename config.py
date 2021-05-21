
class Config:
    FREQ_LBOUND = 5
    PAD = 0
    MAX_LEN = 24
    MAX_N_SEQ = MAX_LEN*2 + 1
    VAL_RATIO = 0.05
    N_HEAD = 4
    N_LAYER = 2
    LR = 1e-4
    EPS = 1e-6
    MAX_DECAY_STEP = 1000000
    DECAY_STEP = 100
    WARMUP_PERIOD = 100
    N_EPOCHS = 1
    TRAIN_BATCH_SIZE = 128
    VAL_BATCH_SIZE = 64
    MAX_N_SHOT = 10
    PATIENCE = 10