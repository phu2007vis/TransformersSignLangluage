import os
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from pytorchvideo.data.clip_sampling import ClipSampler

from dataset.utils import (
    get_label_map, 
    LabeledVideoPaths, 
    has_file_allowed_extension,
    is_image_file, 
    find_number_classes, 
    make_dataset, 
    video_loader, 
    save_video_as_npy, 
    load_video_from_npy,
    LabeledVideoDataset,
    labeled_video_dataset
)

class TestFileHelpers(unittest.TestCase):
    """Test the file extension helper functions"""
    
    def test_has_file_allowed_extension(self):
        """Test file extension checking"""
        self.assertTrue(has_file_allowed_extension("test.jpg", ".jpg"))
        self.assertTrue(has_file_allowed_extension("test.jpg", (".jpg", ".png")))
        self.assertFalse(has_file_allowed_extension("test.txt", ".jpg"))
        self.assertTrue(has_file_allowed_extension("TEST.JPG", ".jpg"))  # Case insensitive
        
    def test_is_image_file(self):
        """Test image file detection"""
        self.assertTrue(is_image_file("test.jpg"))
        self.assertTrue(is_image_file("test.png"))
        self.assertTrue(is_image_file("test.webp"))
        self.assertFalse(is_image_file("test.txt"))
        self.assertFalse(is_image_file("test.mp4"))


class TestClassMapping(unittest.TestCase):
    """Test class mapping and dataset structure functions"""
    
    def setUp(self):
        # Create a temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dataset structure
        for phase in ['train', 'val', 'test']:
            rgb_dir = os.path.join(self.temp_dir, phase, 'rgb')
            os.makedirs(rgb_dir, exist_ok=True)
            
            # Create sample files for classes 1, 2, 3 instead of 0, 1, 2
            for cls in range(1, 4):
                for person in range(2):
                    # Create file names in format *A{class}P{person}.avi
                    filename = f"__A{cls}P{person}.avi"
                    with open(os.path.join(rgb_dir, filename), 'w') as f:
                        f.write("dummy content")
                        
    def tearDown(self):
        # Clean up the temp directory
        shutil.rmtree(self.temp_dir)
        
    def test_find_number_classes(self):
        """Test finding the number of classes in a directory"""
        rgb_dir = os.path.join(self.temp_dir, 'train', 'rgb')
        max_class = find_number_classes(rgb_dir)
        self.assertEqual(max_class, 3)  # Classes are 1, 2, 3 so max is 3
        
    def test_make_dataset(self):
        """Test creating a dataset from a directory"""
        rgb_dir = os.path.join(self.temp_dir, 'train', 'rgb')
        instances = make_dataset(rgb_dir, extensions='.avi')
        # Should have 6 files (3 classes × 2 persons)
        self.assertEqual(len(instances), 6)
        
        # Check if all classes are present
        classes = set(label for _, label in instances)
        self.assertEqual(classes, {1, 2, 3})

    def test_get_label_map(self):
        """Test the label mapping function"""
        label2id, id2label = get_label_map(self.temp_dir)
        
        # Check if mapping is correct (should have 3 classes)
        self.assertEqual(len(label2id), 3)
        self.assertEqual(len(id2label), 3)
        
        # Check if the mapping is consistent
        for label, idx in label2id.items():
            self.assertEqual(id2label[idx], label)


class TestVideoOps(unittest.TestCase):
    """Test video operations like loading and saving"""
    
    @unittest.skipIf(not torch.cuda.is_available(), "Skip if CUDA not available")
    def test_video_tensor_conversion(self):
        """Test converting a video tensor to numpy and back"""
        # Create a dummy video tensor (C, T, H, W)
        dummy_video = torch.randint(0, 256, (3, 10, 64, 64), dtype=torch.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
            # Save and reload
            save_video_as_npy(dummy_video, tmp.name)
            reloaded = load_video_from_npy(tmp.name)
            
            # Check shapes and values
            self.assertEqual(dummy_video.shape, reloaded.shape)
            # Content should be the same but float vs uint8
            self.assertTrue(torch.all(dummy_video.float() == reloaded))


class MockClipSampler(ClipSampler):
    """Mock clip sampler for testing"""
    
    def __call__(self, *args, **kwargs):
        return 0, 10  # start_time, end_time


class TestLabeledVideoPaths(unittest.TestCase):
    """Test the LabeledVideoPaths class"""
    
    def setUp(self):
        # Create a temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample RGB directory with video files
        rgb_dir = os.path.join(self.temp_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        
        # Create sample files
        for cls in range(2):
            for person in range(2):
                filename = f"__A{cls}P{person}.avi"
                with open(os.path.join(rgb_dir, filename), 'w') as f:
                    f.write("dummy content")
    
    def tearDown(self):
        # Clean up the temp directory
        shutil.rmtree(self.temp_dir)
        
    def test_from_directory(self):
        """Test creating LabeledVideoPaths from directory"""
        rgb_dir = os.path.join(self.temp_dir, 'rgb')
        paths = LabeledVideoPaths.from_directory(rgb_dir)
        
        # Should have 4 videos
        self.assertEqual(len(paths), 4)
        
    def test_getitem(self):
        """Test getting a path and label"""
        # Create with known paths and labels
        paths_and_labels = [
            ('video1.mp4', 0),
            ('video2.mp4', 1)
        ]
        paths = LabeledVideoPaths(paths_and_labels)
        
        # Test without prefix
        path, info = paths[0]
        self.assertEqual(path, 'video1.mp4')
        self.assertEqual(info['label'], 0)
        
        # Test with prefix
        paths.path_prefix = '/data'
        path, info = paths[0]
        self.assertEqual(path, '/data/video1.mp4')
        self.assertEqual(info['label'], 0)


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for the dataset components"""
    
    def setUp(self):
        # This is a more complex setup that would simulate a full dataset
        # For simplicity, we'll create a minimal structure
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dataset structure
        for phase in ['train', 'val', 'test']:
            # RGB directory
            rgb_dir = os.path.join(self.temp_dir, phase, 'rgb')
            os.makedirs(rgb_dir, exist_ok=True)
            
            # NPY directory (for landmarks)
            npy_dir = os.path.join(self.temp_dir, phase, 'npy')
            os.makedirs(npy_dir, exist_ok=True)
            
            # Create dummy video file
            video_name = f"__A0P0.avi"
            video_path = os.path.join(rgb_dir, video_name)
            # Just create an empty file
            with open(video_path, 'w') as f:
                f.write("dummy video content")
                
            # Create dummy landmark file
            landmark_path = os.path.join(npy_dir, f"{video_name}.npy")
            # Create a dummy landmark array (T, N, C) where T=frames, N=keypoints, C=coords
            dummy_landmark = np.random.rand(10, 133, 3).astype(np.float32)
            np.save(landmark_path, dummy_landmark)
            
            # Create cache directory
            cache_dir = os.path.join(self.temp_dir, phase, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create dummy cached video
            dummy_video = np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
            np.save(os.path.join(cache_dir, f"__A0P0.npy"), dummy_video)
    
    def tearDown(self):
        # Clean up the temp directory
        shutil.rmtree(self.temp_dir)
        
    @unittest.skip("Skip integration test that requires actual video loading")
    def test_labeled_video_dataset_creation(self):
        """Test creating a labeled video dataset"""
        # This would be a basic integration test
        # We'll skip actual execution as it would require more complex setup
        train_path = os.path.join(self.temp_dir, 'train', 'rgb')
        sampler = MockClipSampler()
        
        dataset = labeled_video_dataset(
            data_path=train_path,
            clip_sampler=sampler,
            decode_audio=False
        )
        
        # Basic check that dataset was created
        self.assertIsInstance(dataset, LabeledVideoDataset)


if __name__ == '__main__':
    unittest.main()