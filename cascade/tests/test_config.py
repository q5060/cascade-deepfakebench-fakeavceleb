import unittest

from cascade.config import validate_shared_input_contract


def compatible_config():
    return {
        'resolution': 256,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'with_mask': False,
        'with_landmark': False,
        'frame_num': {'test': 32},
    }


class SharedInputContractTest(unittest.TestCase):
    def test_compatible_configs_are_accepted(self):
        validate_shared_input_contract(compatible_config(), compatible_config())

    def test_different_frame_sampling_is_rejected(self):
        stage1 = compatible_config()
        stage2 = compatible_config()
        stage2['frame_num'] = {'test': 16}

        with self.assertRaises(ValueError):
            validate_shared_input_contract(stage1, stage2)


if __name__ == '__main__':
    unittest.main()
