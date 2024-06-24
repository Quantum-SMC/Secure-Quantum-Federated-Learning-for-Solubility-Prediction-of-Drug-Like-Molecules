sweep_config = {
    'method': 'bayes',  # 'grid', 'random' or 'bayes'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'net': {
            'values': ['GCN']
        },
        'task_type': {
            'values': ['regression']
        },
        'dataset': {
            'values': ['ESOL']
        },
        'lr': {
            'values': [0.001, 0.00001, 0.00025]
        },
        'npartv': {
            'values': [2]
        },
        'use_aggregation': {
            'values': [True]
        },
        'server_pc': {
            'values': [100]
        },
        'bias': {
            'values': [0.1, 0.5, 0.9]
        },
        'p': {
            'values': [0.05, 0.1, 0.6]
        },
        'niter': {
            'values': [1]
        },
        'batch_size': {
            'values': [64]
        },
        'gpu': {
            'values': [-1]
        },
        'seed': {
            'values': [1]
        },
        'nruns': {
            'values': [1]
        },
        'test_every': {
            'values': [1]
        },
        'aggregation': {
            'values': ['fltrust']
        },
        'flod_threshold': {
            'values': [0.1, 0.5, 0.9]
        },
        'flame_epsilon': {
            'values': [1000, 3000, 5000]
        },
        'flame_delta': {
            'values': [0.0001, 0.001, 0.01]
        },
        'dnc_niters': {
            'values': [2, 5, 9]
        },
        'dnc_c': {
            'values': [1]
        },
        'dnc_b': {
            'values': [1000, 2000, 4000]
        },
        'nbyz': {
            'values': [2, 6, 8]
        },
        'byz_type': {
            'values': ['no']
        },
        'mpspdz': {
            'values': [True]
        },
        'port': {
            'values': [14000]
        },
        'chunk_size': {
            'values': [50, 200, 400]
        },
        'protocol': {
            'values': ['mascot']
        },
        'players': {
            'values': [2, 10]
        },
        'threads': {
            'values': [1]
        },
        'parallels': {
            'values': [1]
        },
        'always_compile': {
            'values': [False]
        },
        'local_epoch': {
            'values': [50]
        },
        'adam_lr': {
            'values': [0.0007]
        }
    }
}
