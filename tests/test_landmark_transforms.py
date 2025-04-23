import unittest
import torch
import numpy as np
import sys
import os
import torch.nn.functional as F

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.landmark_transforms import (
    RandomHorizontalFlip,
    Normalize,
    UniformTemporalSubsample,
    RandomCrop,
    Resize
)

class TestLandmarkTransforms(unittest.TestCase):
    """Test suite for landmark transformation classes"""

    def setUp(self):
        """Set up test fixtures with sample landmark data"""
        # Create sample landmark data with shape (T, N, C) = (10, 133, 3)
        # T = number of frames, N = number of keypoints, C = coordinates (x, y, z)
        self.frames = 10
        self.keypoints = 133
        self.coords = 3
        
        # Generate random landmark data
        torch.manual_seed(42)  # For reproducibility
        self.landmarks = torch.rand(self.frames, self.keypoints, self.coords) * 200
        # Set some values for easier testing
        # First frame, first keypoint to known value [100, 50, 25]
        self.landmarks[0, 0] = torch.tensor([100.0, 50.0, 25.0])

    def test_random_horizontal_flip_no_flip(self):
        """Test RandomHorizontalFlip with p=0 (no flip)"""
        transform = RandomHorizontalFlip(p=0)
        result = transform(self.landmarks)
        
        # No flip should occur, result should be the same
        self.assertTrue(torch.all(torch.eq(result, self.landmarks)))
        self.assertEqual(result[0, 0, 0].item(), 100.0)

    def test_random_horizontal_flip_with_flip(self):
        """Test RandomHorizontalFlip with p=1 (always flip)"""
        transform = RandomHorizontalFlip(p=1)
        result = transform(self.landmarks)
        
        # X coordinates should be negated
        self.assertFalse(torch.all(torch.eq(result, self.landmarks)))
        self.assertEqual(result[0, 0, 0].item(), -100.0)  # X coordinate negated
        self.assertEqual(result[0, 0, 1].item(), 50.0)    # Y coordinate unchanged
        self.assertEqual(result[0, 0, 2].item(), 25.0)    # Z coordinate unchanged
        
        # Make sure the operation didn't modify the original tensor
        self.assertEqual(self.landmarks[0, 0, 0].item(), 100.0)

    def test_normalize(self):
        """Test Normalize transform"""
        mean = (50, 25, 10)
        std = (100, 100, 100)
        transform = Normalize(mean=mean, std=std)
        
        result = transform(self.landmarks)
        
        # Verify normalization formula was applied: (x - mean) / std
        expected_x = (100 - mean[0]) / std[0]
        expected_y = (50 - mean[1]) / std[1]
        expected_z = (25 - mean[2]) / std[2]
        
        self.assertAlmostEqual(result[0, 0, 0].item(), expected_x, places=6)
        self.assertAlmostEqual(result[0, 0, 1].item(), expected_y, places=6)
        self.assertAlmostEqual(result[0, 0, 2].item(), expected_z, places=6)

    def test_normalize_inplace(self):
        """Test Normalize transform with inplace=True"""
        mean = (50, 25, 10)
        std = (100, 100, 100)
        transform = Normalize(mean=mean, std=std, inplace=True)
        
        # Store original memory location
        original_storage_ptr = self.landmarks.storage().data_ptr()
        
        result = transform(self.landmarks)
        
        # Verify it's the same object (inplace)
        self.assertEqual(result.storage().data_ptr(), original_storage_ptr)

    def test_uniform_temporal_subsample(self):
        """Test UniformTemporalSubsample"""
        num_samples = 5
        transform = UniformTemporalSubsample(num_samples)
        
        result = transform(self.landmarks)
        
        # Check shape
        self.assertEqual(result.shape, (num_samples, self.keypoints, self.coords))
        
        # Check if frames were sampled at expected indices
        # For 10 frames -> 5 samples, indices should be [0, 2, 4, 7, 9]
        expected_indices = torch.tensor([0, 2, 4, 7, 9])
        for i, idx in enumerate(expected_indices):
            print(f"Checking index {i} against expected index {idx}")
            self.assertTrue(torch.allclose(result[i], self.landmarks[idx]))

    def test_random_crop(self):
        """Test RandomCrop transform"""
        # Mock torch.randint to always return the same values
        original_randint = torch.randint
        torch.randint = lambda *args, **kwargs: torch.tensor([20])
        
        try:
            output_size = (100, 120)
            original_size = (256, 256)
            transform = RandomCrop(output_size=output_size, original_size=original_size)
            
            result = transform(self.landmarks)
            
            # Check that coordinates were properly transformed:
            # X coordinate: (x - j) * (tw / w) = (100 - 20) * (120 / 256) = 80 * 0.46875 = 37.5
            # Y coordinate: (y - i) * (th / h) = (50 - 20) * (100 / 256) = 30 * 0.390625 = 11.71875
            self.assertAlmostEqual(result[0, 0, 0].item(), 37.5, places=5)
            self.assertAlmostEqual(result[0, 0, 1].item(), 11.71875, places=5)
            # Z coordinate should remain unchanged
            self.assertEqual(result[0, 0, 2].item(), 25.0)
            
        finally:
            # Restore the original function
            torch.randint = original_randint

    def test_resize(self):
        """Test Resize transform"""
        output_size = (112, 112)
        original_size = (256, 256)
        transform = Resize(size=output_size, original_size=original_size)
        
        result = transform(self.landmarks)
        
        # Check that coordinates were properly scaled:
        # X coordinate: x * (w_new / w_orig) = 100 * (112 / 256) = 100 * 0.4375 = 43.75
        # Y coordinate: y * (h_new / h_orig) = 50 * (112 / 256) = 50 * 0.4375 = 21.875
        self.assertAlmostEqual(result[0, 0, 0].item(), 43.75, places=5)
        self.assertAlmostEqual(result[0, 0, 1].item(), 21.875, places=5)
        # Z coordinate should remain unchanged
        self.assertEqual(result[0, 0, 2].item(), 25.0)

    def test_resize_single_number(self):
        """Test Resize transform when size is a single number"""
        size = 112
        original_size = (256, 256)
        transform = Resize(size=size, original_size=original_size)
        
        result = transform(self.landmarks)
        
        # Check that coordinates were properly scaled (same as above since size is square)
        self.assertAlmostEqual(result[0, 0, 0].item(), 43.75, places=5)
        self.assertAlmostEqual(result[0, 0, 1].item(), 21.875, places=5)

    def test_transform_composition(self):
        """Test composing multiple transforms together"""
        from torchvision.transforms import Compose
        
        transforms = Compose([
            UniformTemporalSubsample(num_samples=5),
            Normalize(mean=(50, 25, 10), std=(100, 100, 100)),
            RandomHorizontalFlip(p=1)  # Always flip for testing
        ])
        
        result = transforms(self.landmarks)
        
        # Check shape
        self.assertEqual(result.shape, (5, self.keypoints, self.coords))
        
        # First verify normalize + flip effect on first frame [0, 0]
        # Normalize: ([100, 50, 25] - [50, 25, 10]) / [100, 100, 100] = [0.5, 0.25, 0.15]
        # Then flip X: [-0.5, 0.25, 0.15]
        expected_x = -0.5
        expected_y = 0.25
        expected_z = 0.15
        
        self.assertAlmostEqual(result[0, 0, 0].item(), expected_x, places=5)
        self.assertAlmostEqual(result[0, 0, 1].item(), expected_y, places=5)
        self.assertAlmostEqual(result[0, 0, 2].item(), expected_z, places=5)

if __name__ == '__main__':
    unittest.main()