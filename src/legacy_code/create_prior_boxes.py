def create_prior_boxes_v2(input_size=300,
                scale_min=.20, scale_max=.88,
                num_feature_maps=4,
                feature_map_sizes=[38, 19, 10, 5, 3, 1],
                variances=[.1 ,.2]):

            'min_dim' : 300,
            'steps' : [8, 16, 32, 64, 100, 300],
            'min_sizes' : [30, 60, 111, 162, 213, 264],
            'max_sizes' : [60, 111, 162, 213, 264, 315],
            'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'name' : 'v2'}
    num_feature_maps = 4
    layer_scales = []
    scale = scale_min
    for k in range(1, num_feature_maps + 3):
        scale = scale + ((scale_max - scale_min) / num_feature_maps * (k - 1))
        layer_scales.append(scale)
    layer_scales = [input_size * .1] + layer_scales
    layer_scales = np.asarray(layer_scales)

    for feature_map_size in feature_map_sizes:
        i = list(range(0, feature_map_size))
        j = list(range(0, feature_map_size))
        center_x = (i + 0.5) / feature_map_size
        center_y = (j + 0.5) / feature_map_size


