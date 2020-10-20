from resnet_2d3d import *


def select_resnet(network, track_running_stats=True, distance='dot'):
    param = {'feature_size': 1024}

    # Add CVAE parameters as well (shouldn't hurt existing code)
    # NOTE: unused as of April 2020 (hardcoded parameters because different variants)
    # Fully connected (= on the high side)
    # param['cvae_latent_size_fc'] = 256
    # param['cvae_hidden_size_fc'] = 1024
    # # Convolutional
    # param['cvae_latent_size_conv'] = 64
    # param['cvae_hidden_size_conv'] = 128

    if network == 'resnet8':
        model = resnet8_2d3d_mini(track_running_stats=track_running_stats, distance=distance)
        param['feature_size'] = 16
    elif network == 'resnet10':
        model = resnet10_2d3d_mini(track_running_stats=track_running_stats, distance=distance)
        param['feature_size'] = 16
    elif network == 'resnet18':
        model = resnet18_2d3d_full(track_running_stats=track_running_stats, distance=distance)
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=track_running_stats, distance=distance)
        param['feature_size'] = 256
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=track_running_stats, distance=distance)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(track_running_stats=track_running_stats, distance=distance)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(track_running_stats=track_running_stats, distance=distance)
    elif network == 'resnet200':
        model = resnet200_2d3d_full(track_running_stats=track_running_stats, distance=distance)
    else:
        raise IOError('model type is wrong')

    return model, param
