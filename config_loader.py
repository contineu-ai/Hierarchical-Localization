"""
Configuration loader for COLMAP pipeline
Loads config from YAML and allows CLI overrides
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model paths and configurations"""
    xfeat_weights: str = "weights/xfeat_perm_steer.pth"
    yolo_model: str = "yolo11n.pt"
    lightglue_checkpoint: str = ""
    xfeat_module_path: str = "/data/sahil/new_colmap/Xfeat"
    xfeat_onnx_path: str = "/data/sahil/new_colmap/repo/accelerated_features/model.onnx"
    xfeat_trt_path: str = ""  # TensorRT engine for XFeat
    lightglue_onnx_path: str = ""
    lightglue_trt_path: str = "/data/sahil/new_colmap/repo/LightGlue-ONNX/models/lightglue_spherical.engine"
    rtdetr_onnx_path: str = ""
    rtdetr_trt_path: str = ""  # TensorRT engine for RT-DETR


@dataclass
class FeatureConfig:
    """Feature extraction settings"""
    num_features: int = 3072
    max_keypoints: int = 4096
    detection_threshold: float = 0.05
    top_k: int = 4096


@dataclass
class HumanDetectionConfig:
    """Human detection and masking settings"""
    enabled: bool = True
    confidence_threshold: float = 0.1
    mask_type: str = "solid_color"
    mask_color: list = field(default_factory=lambda: [0, 0, 0])


@dataclass
class PairingConfig:
    """Image pairing settings"""
    num_matches: int = 6


@dataclass
class MatchingConfig:
    """Feature matching settings"""
    min_confidence: float = 0.2
    min_cossim: float = 0.82
    use_spherical_coords: bool = True


@dataclass
class ColmapMapperConfig:
    """COLMAP mapper parameters"""
    ba_gpu_index: int = 0
    ba_use_gpu: int = 1
    ba_refine_focal_length: int = 0
    ba_refine_extra_params: int = 0
    abs_pose_max_error: float = 2.5
    max_reg_trials: int = 6
    tri_min_angle: float = 4.0


@dataclass
class ColmapCameraConfig:
    """COLMAP camera parameters"""
    model_id: int = 12
    focal_multiplier: float = 1.2


@dataclass
class ColmapConfig:
    """COLMAP settings"""
    mapper: ColmapMapperConfig = field(default_factory=ColmapMapperConfig)
    camera: ColmapCameraConfig = field(default_factory=ColmapCameraConfig)


@dataclass
class GPUConfig:
    """GPU and performance settings"""
    device: str = "cuda:0"
    max_memory_gb: float = 20.0
    use_half_precision: bool = True
    compile_models: bool = False
    use_tensorrt: bool = True  # Enable TensorRT acceleration
    tensorrt_fp16: bool = True  # Use FP16 precision in TensorRT
    tensorrt_workspace_gb: float = 4.0  # TensorRT workspace size


@dataclass
class BatchingConfig:
    """Batch processing settings"""
    feature_batch_size: int = 64
    matching_batch_size: int = 16
    num_workers: int = 8
    prefetch_batches: int = 2


@dataclass
class IOConfig:
    """I/O settings"""
    keep_intermediate: bool = False
    compression_level: int = 4


@dataclass
class LoggingConfig:
    """Logging settings"""
    verbose: bool = True
    progress_bars: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    models: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    human_detection: HumanDetectionConfig = field(default_factory=HumanDetectionConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    io: IOConfig = field(default_factory=IOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigLoader:
    """Utility to load and manage configuration"""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def dict_to_dataclass(data: Dict[str, Any], cls):
        """Recursively convert dict to dataclass"""
        if data is None:
            return cls()
        
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in data:
                value = data[field_name]
                
                # Handle nested dataclasses
                if hasattr(field_type, '__dataclass_fields__'):
                    kwargs[field_name] = ConfigLoader.dict_to_dataclass(value, field_type)
                else:
                    kwargs[field_name] = value
        
        return cls(**kwargs)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> PipelineConfig:
        """
        Load configuration from YAML file or use defaults
        
        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        
        Returns:
            PipelineConfig object
        """
        if config_path is None or not Path(config_path).exists():
            print(f"Config file not found, using defaults")
            return PipelineConfig()
        
        print(f"Loading config from: {config_path}")
        yaml_data = cls.load_yaml(config_path)
        config = cls.dict_to_dataclass(yaml_data, PipelineConfig)
        
        return config
    
    @staticmethod
    def override_from_args(config: PipelineConfig, args) -> PipelineConfig:
        """
        Override config values from command-line arguments
        
        Args:
            config: PipelineConfig object
            args: argparse Namespace
        
        Returns:
            Updated PipelineConfig
        """
        # Feature settings
        if hasattr(args, 'num_features') and args.num_features:
            config.features.num_features = args.num_features
        if hasattr(args, 'max_keypoints') and args.max_keypoints:
            config.features.max_keypoints = args.max_keypoints
        
        # Batch settings
        if hasattr(args, 'batch_size') and args.batch_size:
            config.batching.feature_batch_size = args.batch_size
        if hasattr(args, 'matching_batch_size') and args.matching_batch_size:
            config.batching.matching_batch_size = args.matching_batch_size
        if hasattr(args, 'num_workers') and args.num_workers:
            config.batching.num_workers = args.num_workers
        
        # GPU settings
        if hasattr(args, 'gpu_memory') and args.gpu_memory:
            config.gpu.max_memory_gb = args.gpu_memory
        if hasattr(args, 'no_half_precision') and args.no_half_precision:
            config.gpu.use_half_precision = False
        if hasattr(args, 'device') and args.device:
            config.gpu.device = args.device
        if hasattr(args, 'use_tensorrt') and args.use_tensorrt:
            config.gpu.use_tensorrt = True
        if hasattr(args, 'tensorrt_fp16') and args.tensorrt_fp16 is not None:
            config.gpu.tensorrt_fp16 = args.tensorrt_fp16
        
        # Human detection
        if hasattr(args, 'yolo_model') and args.yolo_model:
            config.models.yolo_model = args.yolo_model
        if hasattr(args, 'yolo_conf_threshold') and args.yolo_conf_threshold:
            config.human_detection.confidence_threshold = args.yolo_conf_threshold
        if hasattr(args, 'mask_type') and args.mask_type:
            config.human_detection.mask_type = args.mask_type
        if hasattr(args, 'mask_color') and args.mask_color:
            config.human_detection.mask_color = args.mask_color
        
        # Pairing
        if hasattr(args, 'num_matches') and args.num_matches:
            config.pairing.num_matches = args.num_matches
        
        # Matching
        if hasattr(args, 'min_conf') and args.min_conf is not None:
            config.matching.min_confidence = args.min_conf
        
        # I/O
        if hasattr(args, 'keep_intermediate') and args.keep_intermediate:
            config.io.keep_intermediate = True
        
        # Model paths
        if hasattr(args, 'trt_path') and args.trt_path:
            config.models.lightglue_trt_path = args.trt_path
        if hasattr(args, 'xfeat_trt_path') and args.xfeat_trt_path:
            config.models.xfeat_trt_path = args.xfeat_trt_path
        if hasattr(args, 'rtdetr_trt_path') and args.rtdetr_trt_path:
            config.models.rtdetr_trt_path = args.rtdetr_trt_path
        
        return config
    
    @staticmethod
    def print_config(config: PipelineConfig):
        """Pretty print configuration"""
        print("\n" + "="*60)
        print("PIPELINE CONFIGURATION")
        print("="*60)
        
        print("\n[Models]")
        print(f"  YOLO Model: {config.models.yolo_model}")
        print(f"  XFeat Weights: {config.models.xfeat_weights}")
        if config.models.xfeat_trt_path:
            print(f"  XFeat TRT: {config.models.xfeat_trt_path}")
        if config.models.rtdetr_trt_path:
            print(f"  RT-DETR TRT: {config.models.rtdetr_trt_path}")
        if config.models.lightglue_trt_path:
            print(f"  LightGlue TRT: {config.models.lightglue_trt_path}")
        
        print("\n[Features]")
        print(f"  Num Features: {config.features.num_features}")
        print(f"  Max Keypoints: {config.features.max_keypoints}")
        
        print("\n[Human Detection]")
        print(f"  Enabled: {config.human_detection.enabled}")
        print(f"  Confidence: {config.human_detection.confidence_threshold}")
        print(f"  Mask Type: {config.human_detection.mask_type}")
        
        print("\n[Matching]")
        print(f"  Min Confidence: {config.matching.min_confidence}")
        print(f"  Spherical Coords: {config.matching.use_spherical_coords}")
        
        print("\n[GPU]")
        print(f"  Device: {config.gpu.device}")
        print(f"  Max Memory: {config.gpu.max_memory_gb}GB")
        print(f"  Half Precision: {config.gpu.use_half_precision}")
        print(f"  TensorRT: {config.gpu.use_tensorrt}")
        if config.gpu.use_tensorrt:
            print(f"  TensorRT FP16: {config.gpu.tensorrt_fp16}")
            print(f"  TensorRT Workspace: {config.gpu.tensorrt_workspace_gb}GB")
        
        print("\n[Batching]")
        print(f"  Feature Batch: {config.batching.feature_batch_size}")
        print(f"  Matching Batch: {config.batching.matching_batch_size}")
        print(f"  Workers: {config.batching.num_workers}")
        
        print("\n[COLMAP]")
        print(f"  Camera Model: {config.colmap.camera.model_id}")
        print(f"  BA GPU: {config.colmap.mapper.ba_use_gpu}")
        
        print("="*60 + "\n")


# Convenience function
def load_config(config_path: Optional[str] = None, args=None) -> PipelineConfig:
    """
    Load configuration with optional CLI overrides
    
    Args:
        config_path: Path to YAML config file
        args: argparse Namespace for overrides
    
    Returns:
        PipelineConfig object
    """
    config = ConfigLoader.load(config_path)
    
    if args is not None:
        config = ConfigLoader.override_from_args(config, args)
    
    return config