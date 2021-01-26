from . import sbcnn
from . import strided
from . import ldcnn, dcnn
from . import custom_cnn
from . import reccurent

families = {
    'sbcnn': sbcnn.build_model,
    'ldcnn': ldcnn.ldcnn_nodelta,
    'dcnn': dcnn.dcnn_nodelta,
    'strided': strided.build_model,
    'customcnn': custom_cnn.build_model,
    'recurrent': reccurent.build_model
}


def build(settings):
    builder = families.get(settings['model'])

    options = dict(
        frames=settings['frames'],
        bands=settings['n_mels'],
        channels=settings.get('channels', 1),
    )

    known_settings = [
        'conv_size',
        'conv_block',
        'downsample_size',
        'n_stages',
        'dropout',
        'fully_connected',
        'n_blocks_per_stage',
        'filters',
        'L',
        'W',
        'use_strides'
    ]
    for k in known_settings:
        v = settings.get(k, None)
        options[k] = v

    model = builder(**options)
    return model
