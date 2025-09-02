### Task parameters
DATA_DIR = "/home/ubuntu/IMETA_LAB/data_collection/data"
TASK_CONFIGS = {
    'piper_pick_and_place':{
        'dataset_dir': DATA_DIR + '/piper_place_and_place_0729',
        'episode_len': 800,
        'num_episodes': 60,
        'camera_names': ['cam_right_wrist', "cam_front"],
        "state_dim": 7,
    },
    'piper_pick_and_place_0805':{
        'dataset_dir': DATA_DIR + '/piper_place_and_place_0805',
        'episode_len': 800,
        'num_episodes': 99,
        'camera_names': ['cam_right_wrist', "cam_front"],
        "state_dim": 7,
    },
    'y1_place_and_place_0827':{
        'dataset_dir': DATA_DIR + '/y1_place_and_place_0827',
        'episode_len': 300,
        'num_episodes': 60,
        'camera_names': ["cam_front"],
        "state_dim": 7,
    },
        'y1_place_and_place_0828':{
        'dataset_dir': DATA_DIR + '/y1_place_and_place_0828',
        'episode_len': 300,
        'num_episodes': 159,
        'camera_names': ["cam_front"],
        "state_dim": 7,
    },    
}