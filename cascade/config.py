"""Configuration checks for detectors sharing one sampled-frame dataset."""


SHARED_INPUT_FIELDS = (
    'resolution',
    'mean',
    'std',
    'with_mask',
    'with_landmark',
    'video_mode',
    'clip_size',
)


def validate_shared_input_contract(stage1_config, stage2_config):
    """Ensure both frame detectors can consume the exact same samples."""
    mismatches = []
    for field in SHARED_INPUT_FIELDS:
        stage1_value = stage1_config.get(field)
        stage2_value = stage2_config.get(field)
        if stage1_value != stage2_value:
            mismatches.append(
                '{}: {!r} != {!r}'.format(
                    field, stage1_value, stage2_value
                )
            )

    stage1_frame_num = stage1_config.get('frame_num', {}).get('test')
    stage2_frame_num = stage2_config.get('frame_num', {}).get('test')
    if stage1_frame_num != stage2_frame_num:
        mismatches.append(
            'frame_num.test: {!r} != {!r}'.format(
                stage1_frame_num, stage2_frame_num
            )
        )

    if mismatches:
        raise ValueError(
            'Cascade stages cannot share sampled frames because their input '
            'configs differ: {}.'.format('; '.join(mismatches))
        )
