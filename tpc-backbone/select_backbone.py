from resnet_2d3d import *


def select_resnet(network,
                  track_running_stats=True,
                  distance_type='certain',
                  radius_type='linear'):
    param = {'feature_size': 1024}
    if network == 'resnet8':
        model = resnet8_2d3d_mini(track_running_stats=track_running_stats,
                                  distance_type=distance_type,
                                  radius_type=radius_type)
        if distance_type == 'uncertain':
            param['feature_size'] = 17
        elif distance_type == 'certain':
            param['feature_size'] = 16
    elif network == 'resnet10':
        model = resnet10_2d3d_mini(track_running_stats=track_running_stats,
                                   distance_type=distance_type,
                                   radius_type=radius_type)
        if distance_type == 'uncertain':
            param['feature_size'] = 17
        elif distance_type == 'certain':
            param['feature_size'] = 16
    elif network == 'resnet18':
        model = resnet18_2d3d_full(
            track_running_stats=track_running_stats,
            distance_type=distance_type,
            radius_type=radius_type)
        if distance_type == 'uncertain':
            param['feature_size'] = 257
        elif distance_type == 'certain':
            param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(
            track_running_stats=track_running_stats,
            distance_type=distance_type,
            radius_type=radius_type)
        if distance_type == 'uncertain':
            param['feature_size'] = 257
        elif distance_type == 'certain':
            param['feature_size'] = 256
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet200':
        model = resnet200_2d3d_full(track_running_stats=track_running_stats)
    else:
        raise IOError('model type is wrong')

    return model, param
