### Task parameters
DATA_DIR = "/home/imeta/IMETA_LAB/data_collection/data"
TASK_CONFIGS = {
        'y1_place_and_place_1008':{
        'dataset_dir': DATA_DIR + '/y1_place_and_place_1008',
        'num_episodes': 159,
        'camera_names': ["cam_front"],
        "state_dim": 7,
        },
        'pick_and_place_1009':{
        'dataset_dir': DATA_DIR + '/pick_and_place_1009',
        'num_episodes': 100,
        'camera_names': ["cam_front"],
        "state_dim": 7,
        },    
}