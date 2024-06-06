config = {
    'exp_name': 'lego',
    'log_dir': './output/nerf_synthetic',
    'data_dir': '/root/workdir/dataset/nerf_synthetic/lego',
    
    # Scene settings
    'radius': 1.5,
    
    # Optimization settings
    'num_steps': 20000,
    'batch_size': 8192,
    'lr': 0.01,
    'weight_decay': 1e-6,
    'save_every': 5000,
    'valid_every': 5000,
    
    # Ray sampler settings
    'grid_resolution': 128,
    'grid_nlvl': 1,
    'render_step_size': 5e-3,
    'near_plane': 0.0,
    
    # Model settings
    'model_type': 'NGP',
}
