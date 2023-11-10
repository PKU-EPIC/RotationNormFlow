import sys
import torch
import unittest

from train import main

class TestMainFunction(unittest.TestCase):

    def test_losses_nll_quantile(self):
        # Simulate command-line arguments
        sys.argv = ['train.py', '--config=settings/modelnet_uni.yml', '--max_iteration=500']

        # Call the main function and get the result
        losses_nll = main()

        # Calculate the quantile
        quantile_value = torch.quantile(torch.tensor(losses_nll), 0.4)

        # Check if the quantile value is less than -1.0
        self.assertLess(quantile_value, -1.0, "The 40th percentile of losses_nll should be less than -1.0")

if __name__ == '__main__':
    unittest.main()