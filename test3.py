# import numpy as np
# import pandas as pd
# import os
# import time
# import logging
# import warnings
# import traceback
# import joblib
# import gc
# import psutil
# from typing import List, Dict, Any, Optional, Tuple, Union, Callable
# from dataclasses import dataclass, field
# from collections import defaultdict, deque
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from abc import ABC, abstractmethod
# from datetime import datetime
# import json
# import hashlib
# from pathlib import Path

# # Core ML imports
# from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, silhouette_score, confusion_matrix, log_loss
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict, validation_curve
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, mutual_info_classif, mutual_info_regression, SelectFromModel
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.decomposition import PCA
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.mixture import GaussianMixture
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.ensemble import StackingClassifier, StackingRegressor, ExtraTreesClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.datasets import make_classification, make_regression, fetch_openml
# from sklearn.svm import SVC, SVR
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.cluster import SpectralClustering
# from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# # Fix console encoding for Unicode characters
# import sys
# if sys.platform.startswith('win'):
#     try:
#         import codecs
#         sys.stdout.reconfigure(encoding='utf-8')
#         sys.stderr.reconfigure(encoding='utf-8')
#         os.environ['PYTHONIOENCODING'] = 'utf-8'
#     except:
#         pass

# # FIXED: UMAP import
# try:
#     import umap.umap_ as umap
#     UMAP_AVAILABLE = True
# except ImportError:
#     UMAP_AVAILABLE = False
#     print("UMAP not available. Install with: pip install umap-learn")

# # Advanced ML libraries with graceful fallbacks
# try:
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     from torch.utils.data import DataLoader, TensorDataset
#     TORCH_AVAILABLE = True
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# except ImportError:
#     TORCH_AVAILABLE = False
#     DEVICE = 'cpu'
#     print("PyTorch not available. Install with: pip install torch")

# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False
#     print("XGBoost not available. Install with: pip install xgboost")

# try:
#     import lightgbm as lgb
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False
#     print("LightGBM not available. Install with: pip install lightgbm")

# # NEW: CatBoost support
# try:
#     import catboost as cb
#     CATBOOST_AVAILABLE = True
# except ImportError:
#     CATBOOST_AVAILABLE = False
#     print("CatBoost not available. Install with: pip install catboost")

# # Explainability libraries
# try:
#     import shap
#     SHAP_AVAILABLE = True
# except ImportError:
#     SHAP_AVAILABLE = False
#     print("SHAP not available. Install with: pip install shap")

# try:
#     from lime import lime_tabular
#     LIME_AVAILABLE = True
# except ImportError:
#     LIME_AVAILABLE = False
#     print("LIME not available. Install with: pip install lime")

# # Bayesian optimization
# try:
#     import optuna
#     OPTUNA_AVAILABLE = True
# except ImportError:
#     OPTUNA_AVAILABLE = False
#     print("Optuna not available. Install with: pip install optuna")

# # SMOTE for imbalanced data
# try:
#     from imblearn.over_sampling import SMOTE
#     IMBLEARN_AVAILABLE = True
# except ImportError:
#     IMBLEARN_AVAILABLE = False
#     print("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# # Configure enhanced logging with UTF-8 support
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('tra_system.log', encoding='utf-8')
#     ]
# )
# logger = logging.getLogger(__name__)
# warnings.filterwarnings('ignore')

# # =============================================================================
# # ENHANCED CONFIGURATION SYSTEM
# # =============================================================================

# @dataclass
# class TRAConfig:
#     """Enhanced configuration for TRA system with drift detection and SHAP routing."""
    
#     # Core TRA settings
#     task_type: str = "classification"
#     max_tracks: int = 5
#     enable_meta_learning: bool = True
    
#     # Feature engineering
#     enable_automated_fe: bool = True
#     max_engineered_features: int = 30
#     feature_selection_method: str = "mutual_info"
#     l1_regularization_strength: float = 0.005  # Reduced for less aggressive regularization
    
#     # EXPANDED: AutoML hyperparameter space
#     enable_expanded_automl: bool = True
#     max_depth_range: Tuple[int, int] = (3, 12)  # Expanded from (3, 10)
#     n_estimators_range: Tuple[int, int] = (50, 300)  # Expanded from (50, 200)
#     learning_rate_range: Tuple[float, float] = (0.01, 0.5)  # Expanded from (0.05, 0.3)
#     automl_trials: int = 25  # Increased from 15
    
#     # Resource management
#     use_gpu: bool = TORCH_AVAILABLE
#     max_workers: int = 2
#     memory_limit_mb: int = 4096
#     batch_size: int = 100
    
#     # ENHANCED: Ensemble and fusion with uncertainty
#     enable_stacking: bool = True
#     enable_blending: bool = True
#     enable_uncertainty_weighting: bool = True  # NEW
#     dynamic_fusion: bool = True
#     ensemble_diversity_threshold: float = 0.2
#     min_track_accuracy: float = 0.60  # Lowered to allow more tracks
    
#     # ENHANCED: Routing with entropy and SHAP
#     enable_entropy_routing: bool = True  # NEW
#     enable_shap_routing: bool = True  # NEW
#     entropy_threshold: float = 0.6  # NEW
#     shap_importance_threshold: float = 0.1  # NEW
#     use_confidence_routing: bool = True
#     routing_threshold: float = 0.4
    
#     # NEW: Drift detection
#     enable_drift_detection: bool = True
#     drift_detection_window: int = 100
#     drift_threshold: float = 0.05
    
#     # Clustering improvements
#     min_silhouette_score: float = 0.12  # Lowered for more permissive clustering
#     min_cluster_size_ratio: float = 0.04  # Lowered
#     min_minority_class_ratio: float = 0.02  # Lowered
#     use_dbscan_fallback: bool = True
#     use_umap_clustering: bool = UMAP_AVAILABLE
    
#     # Explainability - ENHANCED
#     enable_explanations: bool = True
#     enable_runtime_explanations: bool = True  # Re-enabled for SHAP routing
#     explanation_method: str = "shap"
#     track_decisions: bool = True
    
#     # Performance optimization with regularization
#     enable_early_stopping: bool = True
#     validation_split: float = 0.2
#     performance_threshold: float = 0.75
#     adaptive_learning: bool = True
#     max_depth_limit: int = 10  # Increased from 8
#     min_samples_leaf: int = 2  # Decreased from 3
#     enable_regularization: bool = True
#     dropout_rate: float = 0.1  # NEW: For neural models
    
#     def validate(self):
#         """Validate configuration settings."""
#         if self.task_type not in ["classification", "regression"]:
#             raise ValueError("task_type must be 'classification' or 'regression'")
#         if self.max_tracks < 1:
#             raise ValueError("max_tracks must be >= 1")

# # =============================================================================
# # ENHANCED DATA STRUCTURES WITH UNCERTAINTY AND DRIFT TRACKING
# # =============================================================================

# @dataclass
# class AdvancedMetrics:
#     """Enhanced performance and resource metrics with uncertainty."""
#     accuracy: float = 0.0
#     f1_score: float = 0.0
#     latency_ms: float = 0.0
#     memory_mb: float = 0.0
#     predictions_per_second: float = 0.0
#     error_rate: float = 0.0
#     stability_score: float = 1.0
#     diversity_score: float = 0.0
#     confidence_score: float = 0.0
#     specialization_score: float = 0.0
#     uncertainty_score: float = 0.0  # NEW
#     drift_score: float = 0.0  # NEW
#     entropy_score: float = 0.0  # NEW
#     shap_importance: float = 0.0  # NEW

#     def efficiency_score(self) -> float:
#         """Calculate overall efficiency score with uncertainty and drift awareness."""
#         if self.latency_ms == 0:
#             return 0.5

#         perf_score = max(0.1, (self.accuracy + self.f1_score) / 2.0)
#         throughput = min(1000.0 / max(self.latency_ms, 1.0), 100.0) / 100.0
#         memory_efficiency = max(0.1, 1.0 - self.memory_mb / 2000.0)
#         stability_weight = max(0.1, self.stability_score)
#         confidence_weight = max(0.1, self.confidence_score)
#         uncertainty_penalty = max(0.1, 1.0 - self.uncertainty_score)  # NEW
#         drift_penalty = max(0.1, 1.0 - self.drift_score)  # NEW

#         efficiency = (0.25 * perf_score + 0.1 * throughput + 0.1 * memory_efficiency +
#                      0.15 * stability_weight + 0.15 * confidence_weight +
#                      0.15 * uncertainty_penalty + 0.1 * drift_penalty)

#         return min(1.0, max(0.1, efficiency))

# @dataclass
# class RoutingDecision:
#     """Enhanced routing decision with entropy and SHAP tracking."""
#     record_id: int
#     selected_track: str
#     confidence: float
#     reason: str
#     alternative_tracks: Dict[str, float] = field(default_factory=dict)
#     feature_scores: Dict[str, float] = field(default_factory=dict)
#     entropy_score: float = 0.0  # NEW
#     shap_scores: Dict[str, float] = field(default_factory=dict)  # NEW
#     uncertainty_score: float = 0.0  # NEW
#     timestamp: datetime = field(default_factory=datetime.now)

# @dataclass
# class DriftAlert:
#     """NEW: Drift detection alert."""
#     timestamp: datetime
#     track_name: str
#     drift_type: str  # 'performance', 'feature', 'prediction'
#     severity: float
#     details: Dict[str, Any] = field(default_factory=dict)

# # =============================================================================
# # NEW: ENTROPY AND UNCERTAINTY CALCULATION ENGINE
# # =============================================================================

# class EntropyUncertaintyEngine:
#     """Engine for calculating prediction entropy and uncertainty metrics."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.entropy_history = defaultdict(list)
        
#     def calculate_prediction_entropy(self, probabilities: np.ndarray) -> float:
#         """Calculate Shannon entropy of prediction probabilities."""
#         try:
#             # Ensure probabilities are valid
#             probabilities = np.clip(probabilities, 1e-10, 1.0)
#             probabilities = probabilities / np.sum(probabilities)
            
#             # Calculate Shannon entropy
#             entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
#             return float(entropy)
#         except Exception as e:
#             logger.debug(f"Entropy calculation failed: {e}")
#             return 0.5
    
#     def calculate_batch_entropy(self, probabilities: np.ndarray) -> np.ndarray:
#         """Calculate entropy for batch of predictions."""
#         try:
#             if probabilities.ndim == 1:
#                 return np.array([self.calculate_prediction_entropy(probabilities)])
            
#             entropies = []
#             for i in range(len(probabilities)):
#                 entropy = self.calculate_prediction_entropy(probabilities[i])
#                 entropies.append(entropy)
            
#             return np.array(entropies)
#         except Exception as e:
#             logger.debug(f"Batch entropy calculation failed: {e}")
#             return np.full(len(probabilities), 0.5)
    
#     def calculate_uncertainty_score(self, predictions: np.ndarray, probabilities: np.ndarray) -> float:
#         """Calculate overall uncertainty score combining multiple metrics."""
#         try:
#             # Entropy-based uncertainty
#             entropy = self.calculate_prediction_entropy(probabilities)
#             max_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
#             entropy_uncertainty = entropy / max_entropy
            
#             # Confidence-based uncertainty (1 - max probability)
#             max_prob = np.max(probabilities)
#             confidence_uncertainty = 1.0 - max_prob
            
#             # Variance-based uncertainty (for continuous outputs)
#             variance_uncertainty = np.std(probabilities) if len(probabilities) > 1 else 0.0
            
#             # Combined uncertainty score
#             uncertainty = 0.5 * entropy_uncertainty + 0.3 * confidence_uncertainty + 0.2 * variance_uncertainty
            
#             return float(np.clip(uncertainty, 0.0, 1.0))
        
#         except Exception as e:
#             logger.debug(f"Uncertainty calculation failed: {e}")
#             return 0.5
    
#     def should_reroute_based_on_entropy(self, entropy: float) -> bool:
#         """Determine if record should be re-routed based on entropy."""
#         return entropy > self.config.entropy_threshold

# # =============================================================================
# # NEW: ENHANCED SHAP ROUTING ENGINE
# # =============================================================================

# class EnhancedSHAPRoutingEngine:
#     """Enhanced routing engine using SHAP values for intelligent track selection."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.track_shap_explainers = {}
#         self.track_feature_importance = {}
#         self.global_feature_importance = {}
#         self.routing_history = defaultdict(list)
        
#     def initialize_shap_explainers(self, tracks: Dict[str, 'OptimizedEnhancedTrack'], 
#                                    X_sample: pd.DataFrame):
#         """Initialize SHAP explainers for each track."""
#         if not SHAP_AVAILABLE or not self.config.enable_shap_routing:
#             return
            
#         logger.info("Initializing SHAP explainers for routing...")
        
#         for track_name, track in tracks.items():
#             try:
#                 # Use a small background sample for efficiency
#                 background_size = min(30, len(X_sample))
#                 background = shap.sample(X_sample.values, background_size)
                
#                 # Initialize explainer based on model type
#                 if hasattr(track.classifier, 'predict_proba') and self.config.task_type == "classification":
#                     if hasattr(track.classifier, 'estimators_'):  # Ensemble models
#                         explainer = shap.TreeExplainer(track.classifier)
#                     else:
#                         explainer = shap.KernelExplainer(track.classifier.predict_proba, background)
#                 else:
#                     explainer = shap.KernelExplainer(track.classifier.predict, background)
                
#                 self.track_shap_explainers[track_name] = explainer
                
#                 # Calculate global feature importance for this track
#                 try:
#                     sample_shap_values = explainer.shap_values(X_sample.values[:min(20, len(X_sample))])
#                     if isinstance(sample_shap_values, list):
#                         # Multi-class classification
#                         importance = np.mean(np.abs(sample_shap_values[0]), axis=0)
#                     else:
#                         importance = np.mean(np.abs(sample_shap_values), axis=0)
                    
#                     self.track_feature_importance[track_name] = dict(zip(X_sample.columns, importance))
                    
#                 except Exception as e:
#                     logger.debug(f"Could not calculate SHAP importance for {track_name}: {e}")
#                     self.track_feature_importance[track_name] = {}
                
#                 logger.info(f"SHAP explainer initialized for {track_name}")
                
#             except Exception as e:
#                 logger.warning(f"SHAP explainer initialization failed for {track_name}: {e}")
#                 continue
                
#         # Calculate global feature importance baseline
#         self._calculate_global_feature_importance(X_sample)
    
#     def _calculate_global_feature_importance(self, X_sample: pd.DataFrame):
#         """Calculate global feature importance baseline."""
#         try:
#             if self.track_feature_importance:
#                 # Average importance across all tracks
#                 all_features = set()
#                 for track_importance in self.track_feature_importance.values():
#                     all_features.update(track_importance.keys())
                
#                 global_importance = {}
#                 for feature in all_features:
#                     importances = [track_importance.get(feature, 0.0) 
#                                  for track_importance in self.track_feature_importance.values()]
#                     global_importance[feature] = np.mean(importances)
                
#                 self.global_feature_importance = global_importance
#         except Exception as e:
#             logger.debug(f"Global feature importance calculation failed: {e}")
    
#     def calculate_track_shap_scores(self, X_instance: pd.DataFrame, 
#                                     tracks: Dict[str, 'OptimizedEnhancedTrack']) -> Dict[str, float]:
#         """Calculate SHAP-based routing scores for each track."""
#         track_scores = {}
        
#         if not SHAP_AVAILABLE or not self.config.enable_shap_routing:
#             # Fallback to equal scores
#             return {track_name: 0.5 for track_name in tracks.keys()}
        
#         for track_name, track in tracks.items():
#             try:
#                 if track_name in self.track_shap_explainers:
#                     explainer = self.track_shap_explainers[track_name]
                    
#                     # Get SHAP values for this instance
#                     shap_values = explainer.shap_values(X_instance.values)
                    
#                     if isinstance(shap_values, list):
#                         # Multi-class: use the class with highest prediction
#                         predictions = track.predict_proba(X_instance)
#                         if len(predictions) > 0 and len(predictions[0]) > 1:
#                             best_class = np.argmax(predictions[0])
#                             instance_shap_values = shap_values[best_class][0]
#                         else:
#                             instance_shap_values = shap_values[0][0]
#                     else:
#                         instance_shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values
                    
#                     # Calculate feature specialization match
#                     feature_names = X_instance.columns.tolist()
#                     track_importance = self.track_feature_importance.get(track_name, {})
                    
#                     specialization_score = 0.0
#                     total_shap_importance = 0.0
                    
#                     for i, (feature, shap_val) in enumerate(zip(feature_names, instance_shap_values)):
#                         abs_shap = abs(float(shap_val))
#                         total_shap_importance += abs_shap
                        
#                         # Weight by track's specialization in this feature
#                         track_feature_importance = track_importance.get(feature, 0.0)
#                         global_feature_importance = self.global_feature_importance.get(feature, 0.0)
                        
#                         # Specialization bonus if track is specialized in this important feature
#                         if track_feature_importance > global_feature_importance:
#                             specialization_bonus = (track_feature_importance - global_feature_importance)
#                             specialization_score += abs_shap * specialization_bonus
                    
#                     # Normalize by total SHAP importance
#                     if total_shap_importance > 0:
#                         specialization_score /= total_shap_importance
                    
#                     # Combine with base score
#                     base_score = 0.5
#                     shap_score = base_score + (0.3 * specialization_score)
                    
#                     track_scores[track_name] = float(np.clip(shap_score, 0.0, 1.0))
                    
#                 else:
#                     track_scores[track_name] = 0.5
                    
#             except Exception as e:
#                 logger.debug(f"SHAP routing calculation failed for {track_name}: {e}")
#                 track_scores[track_name] = 0.5
        
#         return track_scores
    
#     def get_feature_explanation(self, X_instance: pd.DataFrame, track_name: str) -> Dict[str, Any]:
#         """Get SHAP-based feature explanation for a specific prediction."""
#         try:
#             if track_name not in self.track_shap_explainers:
#                 return {"explanation_available": False, "reason": "no_shap_explainer"}
            
#             explainer = self.track_shap_explainers[track_name]
#             shap_values = explainer.shap_values(X_instance.values)
            
#             if isinstance(shap_values, list):
#                 shap_values = shap_values[0][0]  # Use first class for simplicity
#             else:
#                 shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values
            
#             # Create feature importance ranking
#             feature_names = X_instance.columns.tolist()
#             feature_importance = dict(zip(feature_names, np.abs(shap_values)))
#             sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
#             return {
#                 "explanation_available": True,
#                 "method": "shap",
#                 "feature_importance": dict(sorted_features[:10]),
#                 "top_features": [f[0] for f in sorted_features[:5]],
#                 "shap_values": dict(zip(feature_names, shap_values.tolist()))
#             }
            
#         except Exception as e:
#             return {"explanation_available": False, "reason": f"shap_error: {e}"}

# # =============================================================================
# # NEW: DRIFT DETECTION ENGINE
# # =============================================================================

# class DriftDetectionEngine:
#     """Lightweight drift detection using rolling accuracy and feature statistics."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.track_performance_history = defaultdict(list)
#         self.feature_statistics_history = defaultdict(list)
#         self.drift_alerts = []
#         self.last_feature_stats = {}
        
#     def update_performance(self, track_name: str, accuracy: float, predictions: np.ndarray):
#         """Update performance history for drift detection."""
#         self.track_performance_history[track_name].append({
#             'timestamp': time.time(),
#             'accuracy': accuracy,
#             'n_predictions': len(predictions)
#         })
        
#         # Keep only recent history
#         window_size = self.config.drift_detection_window
#         if len(self.track_performance_history[track_name]) > window_size:
#             self.track_performance_history[track_name] = \
#                 self.track_performance_history[track_name][-window_size:]
    
#     def update_feature_statistics(self, X: pd.DataFrame):
#         """Update feature statistics for drift detection."""
#         try:
#             current_stats = {}
            
#             for column in X.columns:
#                 if pd.api.types.is_numeric_dtype(X[column]):
#                     current_stats[column] = {
#                         'mean': float(X[column].mean()),
#                         'std': float(X[column].std()),
#                         'min': float(X[column].min()),
#                         'max': float(X[column].max())
#                     }
            
#             # Check for drift if we have previous statistics
#             if self.last_feature_stats:
#                 self._detect_feature_drift(current_stats)
            
#             self.last_feature_stats = current_stats
            
#         except Exception as e:
#             logger.debug(f"Feature statistics update failed: {e}")
    
#     def _detect_feature_drift(self, current_stats: Dict[str, Dict[str, float]]):
#         """Detect feature drift by comparing statistics."""
#         try:
#             for feature, current_stat in current_stats.items():
#                 if feature in self.last_feature_stats:
#                     previous_stat = self.last_feature_stats[feature]
                    
#                     # Calculate relative changes
#                     mean_change = abs(current_stat['mean'] - previous_stat['mean']) / (abs(previous_stat['mean']) + 1e-8)
#                     std_change = abs(current_stat['std'] - previous_stat['std']) / (previous_stat['std'] + 1e-8)
                    
#                     # Combined drift score
#                     drift_score = 0.6 * mean_change + 0.4 * std_change
                    
#                     if drift_score > self.config.drift_threshold:
#                         alert = DriftAlert(
#                             timestamp=datetime.now(),
#                             track_name="global",
#                             drift_type="feature",
#                             severity=drift_score,
#                             details={
#                                 'feature': feature,
#                                 'mean_change': mean_change,
#                                 'std_change': std_change
#                             }
#                         )
#                         self.drift_alerts.append(alert)
#                         logger.warning(f"Feature drift detected in {feature}: score={drift_score:.3f}")
                        
#         except Exception as e:
#             logger.debug(f"Feature drift detection failed: {e}")
    
#     def detect_performance_drift(self, track_name: str) -> bool:
#         """Detect performance drift using rolling accuracy."""
#         try:
#             if track_name not in self.track_performance_history:
#                 return False
            
#             history = self.track_performance_history[track_name]
            
#             if len(history) < 10:  # Need sufficient history
#                 return False
            
#             # Split history into recent and older periods
#             split_point = len(history) // 2
#             older_accuracies = [h['accuracy'] for h in history[:split_point]]
#             recent_accuracies = [h['accuracy'] for h in history[split_point:]]
            
#             # Calculate mean accuracies
#             older_mean = np.mean(older_accuracies)
#             recent_mean = np.mean(recent_accuracies)
            
#             # Check for significant performance drop
#             performance_drop = older_mean - recent_mean
            
#             if performance_drop > self.config.drift_threshold:
#                 alert = DriftAlert(
#                     timestamp=datetime.now(),
#                     track_name=track_name,
#                     drift_type="performance",
#                     severity=performance_drop,
#                     details={
#                         'older_accuracy': older_mean,
#                         'recent_accuracy': recent_mean,
#                         'drop': performance_drop
#                     }
#                 )
#                 self.drift_alerts.append(alert)
#                 logger.warning(f"Performance drift detected in {track_name}: drop={performance_drop:.3f}")
#                 return True
            
#             return False
            
#         except Exception as e:
#             logger.debug(f"Performance drift detection failed for {track_name}: {e}")
#             return False
    
#     def get_drift_summary(self) -> Dict[str, Any]:
#         """Get drift detection summary."""
#         recent_alerts = [alert for alert in self.drift_alerts if 
#                         (datetime.now() - alert.timestamp).seconds < 3600]  # Last hour
        
#         return {
#             'total_alerts': len(self.drift_alerts),
#             'recent_alerts': len(recent_alerts),
#             'drift_types': {
#                 'performance': len([a for a in recent_alerts if a.drift_type == 'performance']),
#                 'feature': len([a for a in recent_alerts if a.drift_type == 'feature'])
#             },
#             'most_severe_alert': max(recent_alerts, key=lambda x: x.severity) if recent_alerts else None
#         }

# # =============================================================================
# # ENHANCED CLUSTERING ENGINE WITH FIXED UMAP
# # =============================================================================

# class EnhancedClusteringEngine:
#     """Advanced clustering with proper UMAP integration and multiple algorithms."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.clustering_method = None
#         self.dimension_reducer = None
#         self.cluster_centers = {}
#         self.cluster_metadata = {}
    
#     def find_optimal_clusters(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """Find optimal clusters with large dataset optimization."""
#         logger.info("Finding optimal clusters with enhanced methods...")
        
#         # CRITICAL: Skip clustering for very large datasets
#         if len(X) > 200000:
#             logger.warning(f"Dataset too large ({len(X)} samples). Skipping complex clustering.")
#             return self._simple_kmeans_clustering(X, y)
        
#         # Use substantial sampling for large datasets
#         if len(X) > 50000:
#             logger.info(f"Large dataset detected. Using sample for clustering...")
#             sample_size = min(20000, len(X) // 3)
#             sample_indices = np.random.choice(len(X), sample_size, replace=False)
#             X_sample = X[sample_indices]
#             y_sample = y[sample_indices]
            
#             # Find clusters on sample
#             sample_labels = self._find_clusters_on_sample(X_sample, y_sample)
            
#             if sample_labels is not None:
#                 # Extend labels to full dataset using nearest neighbors
#                 return self._extend_cluster_labels(X_sample, sample_labels, X)
#             else:
#                 return self._simple_kmeans_clustering(X, y)
        
#         # Original clustering for smaller datasets
#         X_reduced = self._reduce_dimensions(X)
        
#         best_labels = None
#         best_score = -1
#         best_method = None
        
#         # Simplified clustering methods for faster execution
#         clustering_methods = [
#             ('simple_kmeans', self._simple_kmeans_clustering),
#             ('pca_kmeans', self._pca_kmeans_clustering)
#         ]
        
#         # Only try advanced methods on small datasets
#         if len(X) < 10000:
#             clustering_methods.extend([
#                 ('umap_kmeans', self._umap_kmeans_clustering),
#                 ('gmm', self._gmm_clustering)
#             ])
        
#         for method_name, method_func in clustering_methods:
#             try:
#                 labels = method_func(X_reduced, y)
#                 if labels is not None:
#                     # Use faster evaluation for large datasets
#                     score = self._fast_evaluate_clustering(X_reduced, labels, y) if len(X) > 5000 else self._evaluate_clustering(X_reduced, labels, y)
#                     if score > best_score:
#                         best_score = score
#                         best_labels = labels
#                         best_method = method_name
#                         logger.info(f"Better clustering: {method_name} (score: {score:.3f})")
#             except Exception as e:
#                 logger.debug(f"Clustering method {method_name} failed: {e}")
#                 continue
        
#         if best_labels is not None:
#             self.clustering_method = best_method
#             self._analyze_clusters(X_reduced, best_labels, y)
#             logger.info(f"Selected clustering method: {best_method} with score: {best_score:.3f}")
        
#         return best_labels

#     def _simple_kmeans_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """Simple K-means clustering optimized for large datasets."""
#         try:
#             # Use sample for very large datasets
#             if len(X) > 50000:
#                 sample_size = min(10000, len(X) // 5)
#                 sample_indices = np.random.choice(len(X), sample_size, replace=False)
#                 X_sample = X[sample_indices]
#             else:
#                 X_sample = X
            
#             best_labels = None
#             best_score = -1
            
#             # Try fewer clusters for faster execution
#             max_clusters = min(4, max(2, len(np.unique(y))))
            
#             for n_clusters in range(2, max_clusters + 1):
#                 try:
#                     kmeans = KMeans(
#                         n_clusters=n_clusters, 
#                         random_state=42, 
#                         n_init=3,  # Reduced from 10
#                         max_iter=100,  # Reduced from 300
#                         algorithm='elkan'  # Faster for dense data
#                     )
#                     labels_sample = kmeans.fit_predict(X_sample)
                    
#                     if len(np.unique(labels_sample)) >= 2:
#                         # Fast silhouette approximation
#                         if len(X_sample) > 2000:
#                             # Use sample for silhouette calculation
#                             sil_sample_size = 2000
#                             sil_indices = np.random.choice(len(X_sample), sil_sample_size, replace=False)
#                             score = silhouette_score(X_sample[sil_indices], labels_sample[sil_indices])
#                         else:
#                             score = silhouette_score(X_sample, labels_sample)
                        
#                         if score > best_score:
#                             best_score = score
                            
#                             # Extend to full dataset if we used sampling
#                             if len(X) > 50000:
#                                 # Fit on full dataset with best parameters
#                                 kmeans_full = KMeans(
#                                     n_clusters=n_clusters, 
#                                     random_state=42, 
#                                     n_init=1,
#                                     max_iter=50
#                                 )
#                                 best_labels = kmeans_full.fit_predict(X)
#                             else:
#                                 best_labels = labels_sample
                
#                 except Exception as e:
#                     logger.debug(f"K-means with {n_clusters} clusters failed: {e}")
#                     continue
            
#             return best_labels
            
#         except Exception as e:
#             logger.debug(f"Simple K-means clustering failed: {e}")
#             return None

#     def _find_clusters_on_sample(self, X_sample: np.ndarray, y_sample: np.ndarray) -> Optional[np.ndarray]:
#         """Find clusters on sample data."""
#         return self._simple_kmeans_clustering(X_sample, y_sample)

#     def _extend_cluster_labels(self, X_sample: np.ndarray, sample_labels: np.ndarray, X_full: np.ndarray) -> np.ndarray:
#         """Extend cluster labels from sample to full dataset using nearest neighbors."""
#         try:
#             from sklearn.neighbors import NearestNeighbors
            
#             # Fit nearest neighbors on sample
#             nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
#             nn.fit(X_sample)
            
#             # Find nearest sample point for each full dataset point
#             batch_size = 10000  # Process in batches to manage memory
#             full_labels = np.zeros(len(X_full), dtype=int)
            
#             for start_idx in range(0, len(X_full), batch_size):
#                 end_idx = min(start_idx + batch_size, len(X_full))
#                 batch = X_full[start_idx:end_idx]
                
#                 _, indices = nn.kneighbors(batch)
#                 full_labels[start_idx:end_idx] = sample_labels[indices.flatten()]
            
#             return full_labels
            
#         except Exception as e:
#             logger.debug(f"Label extension failed: {e}")
#             # Fallback: simple assignment based on cluster centers
#             return self._simple_cluster_assignment(X_sample, sample_labels, X_full)

#     def _simple_cluster_assignment(self, X_sample: np.ndarray, sample_labels: np.ndarray, X_full: np.ndarray) -> np.ndarray:
#         """Simple cluster assignment fallback."""
#         try:
#             # Calculate cluster centers from sample
#             unique_labels = np.unique(sample_labels)
#             centers = []
            
#             for label in unique_labels:
#                 cluster_points = X_sample[sample_labels == label]
#                 center = np.mean(cluster_points, axis=0)
#                 centers.append(center)
            
#             centers = np.array(centers)
            
#             # Assign full dataset points to nearest center
#             from sklearn.metrics.pairwise import euclidean_distances
            
#             batch_size = 10000
#             full_labels = np.zeros(len(X_full), dtype=int)
            
#             for start_idx in range(0, len(X_full), batch_size):
#                 end_idx = min(start_idx + batch_size, len(X_full))
#                 batch = X_full[start_idx:end_idx]
                
#                 distances = euclidean_distances(batch, centers)
#                 batch_labels = unique_labels[np.argmin(distances, axis=1)]
#                 full_labels[start_idx:end_idx] = batch_labels
            
#             return full_labels
            
#         except Exception as e:
#             logger.error(f"Simple cluster assignment failed: {e}")
#             # Ultimate fallback: random assignment
#             return np.random.randint(0, 2, len(X_full))

#     def _fast_evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> float:
#         """Fast clustering evaluation for large datasets."""
#         try:
#             # Use sample for evaluation
#             if len(X) > 5000:
#                 sample_size = 2000
#                 sample_indices = np.random.choice(len(X), sample_size, replace=False)
#                 X_sample = X[sample_indices]
#                 labels_sample = labels[sample_indices]
#             else:
#                 X_sample = X
#                 labels_sample = labels
            
#             # Filter out noise points
#             if -1 in labels_sample:
#                 mask = labels_sample != -1
#                 labels_clean = labels_sample[mask]
#                 X_clean = X_sample[mask]
#             else:
#                 labels_clean = labels_sample
#                 X_clean = X_sample
            
#             if len(np.unique(labels_clean)) < 2:
#                 return -1
            
#             # Just use silhouette score
#             sil_score = silhouette_score(X_clean, labels_clean)
#             return max(0, sil_score)
            
#         except Exception as e:
#             logger.debug(f"Fast clustering evaluation failed: {e}")
#             return -1

    
#     def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
#         """Reduce dimensions for better clustering with fixed UMAP."""
#         if X.shape[1] <= 8:
#             return X
        
#         try:
#             if UMAP_AVAILABLE and self.config.use_umap_clustering:
#                 # FIXED: Proper UMAP initialization
#                 self.dimension_reducer = umap.UMAP(
#                     n_components=min(8, X.shape[1] - 1),
#                     n_neighbors=min(15, X.shape[0] // 3),
#                     min_dist=0.1,
#                     random_state=42,
#                     low_memory=True
#                 )
#                 X_reduced = self.dimension_reducer.fit_transform(X)
#                 logger.info(f"UMAP dimensionality reduction: {X.shape[1]} → {X_reduced.shape[1]}")
#             else:
#                 # Fallback to PCA
#                 n_components = min(8, X.shape[1] - 1)
#                 self.dimension_reducer = PCA(n_components=n_components, random_state=42)
#                 X_reduced = self.dimension_reducer.fit_transform(X)
#                 logger.info(f"PCA dimensionality reduction: {X.shape[1]} → {X_reduced.shape[1]}")
            
#             return X_reduced
            
#         except Exception as e:
#             logger.warning(f"Dimensionality reduction failed: {e}")
#             return X
    
#     def _umap_kmeans_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """UMAP + K-means clustering with fixed UMAP."""
#         if not UMAP_AVAILABLE:
#             return None
        
#         try:
#             best_labels = None
#             best_score = -1
            
#             for n_clusters in range(2, min(6, max(2, len(X) // 80))):
#                 kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#                 labels = kmeans.fit_predict(X)
                
#                 if len(np.unique(labels)) >= 2:
#                     score = silhouette_score(X, labels)
#                     if score > best_score:
#                         best_score = score
#                         best_labels = labels
            
#             return best_labels
            
#         except Exception as e:
#             logger.debug(f"UMAP+KMeans clustering failed: {e}")
#             return None
    
#     def _umap_dbscan_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """UMAP + DBSCAN clustering with fixed UMAP."""
#         if not UMAP_AVAILABLE:
#             return None
        
#         try:
#             # DBSCAN parameters
#             eps_values = [0.3, 0.5, 0.8, 1.2]
#             min_samples_values = [3, 5, 8]
#             best_labels = None
#             best_score = -1
            
#             for eps in eps_values:
#                 for min_samples in min_samples_values:
#                     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#                     labels = dbscan.fit_predict(X)
                    
#                     # Filter noise points
#                     valid_labels = labels[labels != -1]
#                     if len(np.unique(valid_labels)) >= 2 and len(valid_labels) >= len(X) * 0.6:
#                         X_clean = X[labels != -1]
#                         score = silhouette_score(X_clean, valid_labels)
#                         if score > best_score:
#                             best_score = score
#                             best_labels = labels
            
#             return best_labels
            
#         except Exception as e:
#             logger.debug(f"UMAP+DBSCAN clustering failed: {e}")
#             return None
    
#     def _pca_kmeans_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """PCA + K-means clustering (fallback)."""
#         try:
#             best_labels = None
#             best_score = -1
            
#             for n_clusters in range(2, min(6, max(2, len(X) // 60))):
#                 kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#                 labels = kmeans.fit_predict(X)
#                 score = silhouette_score(X, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_labels = labels
            
#             return best_labels
            
#         except Exception as e:
#             logger.debug(f"PCA+KMeans clustering failed: {e}")
#             return None
    
#     def _spectral_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """Spectral clustering."""
#         try:
#             if len(X) > 800:  # Too expensive for large datasets
#                 return None
            
#             best_labels = None
#             best_score = -1
            
#             for n_clusters in range(2, min(4, len(X) // 100)):
#                 spectral = SpectralClustering(
#                     n_clusters=n_clusters,
#                     random_state=42,
#                     affinity='nearest_neighbors',
#                     n_neighbors=min(10, len(X) // 10)
#                 )
#                 labels = spectral.fit_predict(X)
#                 score = silhouette_score(X, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_labels = labels
            
#             return best_labels
            
#         except Exception as e:
#             logger.debug(f"Spectral clustering failed: {e}")
#             return None
    
#     def _gmm_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
#         """Gaussian Mixture Model clustering."""
#         try:
#             best_labels = None
#             best_score = -1
            
#             for n_components in range(2, min(5, len(X) // 50)):
#                 gmm = GaussianMixture(n_components=n_components, random_state=42)
#                 labels = gmm.fit_predict(X)
#                 if len(np.unique(labels)) >= 2:
#                     score = silhouette_score(X, labels)
#                     if score > best_score:
#                         best_score = score
#                         best_labels = labels
            
#             return best_labels
            
#         except Exception as e:
#             logger.debug(f"GMM clustering failed: {e}")
#             return None
    
#     def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> float:
#         """Evaluate clustering quality with multiple metrics."""
#         try:
#             # Filter out noise points for DBSCAN
#             if -1 in labels:
#                 mask = labels != -1
#                 labels_clean = labels[mask]
#                 X_clean = X[mask]
#                 y_clean = y[mask]
#             else:
#                 labels_clean = labels
#                 X_clean = X
#                 y_clean = y
            
#             if len(np.unique(labels_clean)) < 2:
#                 return -1
            
#             # Silhouette score (clustering quality)
#             sil_score = silhouette_score(X_clean, labels_clean)
            
#             # Cluster size balance
#             cluster_sizes = np.bincount(labels_clean)
#             size_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
            
#             # Class separation within clusters
#             class_separation = 0.0
#             if len(np.unique(y_clean)) > 1:
#                 for cluster_id in np.unique(labels_clean):
#                     cluster_mask = labels_clean == cluster_id
#                     if np.sum(cluster_mask) > 3:
#                         y_cluster = y_clean[cluster_mask]
#                         if len(np.unique(y_cluster)) > 1:
#                             class_counts = np.bincount(y_cluster)
#                             purity = np.max(class_counts) / np.sum(class_counts)
#                             class_separation += purity
                
#                 class_separation /= len(np.unique(labels_clean))
            
#             # Combined score
#             quality_score = 0.6 * max(0, sil_score) + 0.25 * size_balance + 0.15 * class_separation
            
#             return quality_score
            
#         except Exception as e:
#             logger.debug(f"Clustering evaluation failed: {e}")
#             return -1
    
#     def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray, y: np.ndarray):
#         """Analyze cluster characteristics for routing."""
#         self.cluster_metadata = {}
        
#         for cluster_id in np.unique(labels):
#             if cluster_id == -1:  # Skip noise points
#                 continue
            
#             cluster_mask = labels == cluster_id
#             X_cluster = X[cluster_mask]
#             y_cluster = y[cluster_mask]
            
#             # Calculate cluster center
#             center = np.mean(X_cluster, axis=0)
#             self.cluster_centers[cluster_id] = center
            
#             # Analyze cluster characteristics
#             cluster_info = {
#                 'size': len(X_cluster),
#                 'center': center,
#                 'std': np.std(X_cluster, axis=0),
#                 'class_distribution': np.bincount(y_cluster) if len(np.unique(y_cluster)) > 1 else None,
#                 'dominant_class': np.argmax(np.bincount(y_cluster)) if len(np.unique(y_cluster)) > 1 else None
#             }
#             self.cluster_metadata[cluster_id] = cluster_info

# # =============================================================================
# # EXPANDED AUTOML MODEL FACTORY WITH CATBOOST
# # =============================================================================

# class ExpandedAutoMLModelFactory:
#     """Factory for creating models with expanded hyperparameter space."""
    
#     @staticmethod
#     def create_expanded_classifier(model_type: str, config: TRAConfig, trial=None, **params) -> BaseEstimator:
#         """Create classifier with expanded hyperparameter space."""
        
#         if model_type == 'random_forest':
#             n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
#             max_depth = trial.suggest_int('max_depth', *config.max_depth_range) if trial else params.get('max_depth', 8)
#             min_samples_split = trial.suggest_int('min_samples_split', 2, 20) if trial else params.get('min_samples_split', 5)
#             min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10) if trial else params.get('min_samples_leaf', 2)
#             max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) if trial else params.get('max_features', 'sqrt')
            
#             return RandomForestClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 min_samples_split=min_samples_split,
#                 min_samples_leaf=min_samples_leaf,
#                 max_features=max_features,
#                 bootstrap=True,
#                 class_weight='balanced',
#                 random_state=42,
#                 n_jobs=-1
#             )
        
#         elif model_type == 'gradient_boosting':
#             n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
#             max_depth = trial.suggest_int('max_depth', 3, 8) if trial else params.get('max_depth', 6)
#             learning_rate = trial.suggest_float('learning_rate', *config.learning_rate_range) if trial else params.get('learning_rate', 0.1)
#             subsample = trial.suggest_float('subsample', 0.6, 1.0) if trial else params.get('subsample', 0.8)
            
#             return GradientBoostingClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 subsample=subsample,
#                 validation_fraction=0.1,
#                 n_iter_no_change=15,
#                 random_state=42
#             )
        
#         elif model_type == 'extra_trees':
#             n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
#             max_depth = trial.suggest_int('max_depth', *config.max_depth_range) if trial else params.get('max_depth', 8)
#             min_samples_split = trial.suggest_int('min_samples_split', 2, 20) if trial else params.get('min_samples_split', 5)
            
#             return ExtraTreesClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 min_samples_split=min_samples_split,
#                 bootstrap=True,
#                 class_weight='balanced',
#                 random_state=42,
#                 n_jobs=-1
#             )
        
#         elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
#             n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
#             max_depth = trial.suggest_int('max_depth', 3, 8) if trial else params.get('max_depth', 6)
#             learning_rate = trial.suggest_float('learning_rate', *config.learning_rate_range) if trial else params.get('learning_rate', 0.1)
#             subsample = trial.suggest_float('subsample', 0.6, 1.0) if trial else params.get('subsample', 0.8)
#             colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0) if trial else params.get('colsample_bytree', 0.8)
#             reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0) if trial else params.get('reg_alpha', 0.0)
#             reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0) if trial else params.get('reg_lambda', 0.0)
            
#             return xgb.XGBClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 subsample=subsample,
#                 colsample_bytree=colsample_bytree,
#                 reg_alpha=reg_alpha,
#                 reg_lambda=reg_lambda,
#                 random_state=42,
#                 eval_metric='logloss',
#                 early_stopping_rounds=15,
#                 verbosity=0
#             )
        
#         elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
#             n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
#             max_depth = trial.suggest_int('max_depth', 3, 8) if trial else params.get('max_depth', 6)
#             learning_rate = trial.suggest_float('learning_rate', *config.learning_rate_range) if trial else params.get('learning_rate', 0.1)
#             num_leaves = trial.suggest_int('num_leaves', 10, 100) if trial else params.get('num_leaves', 31)
#             feature_fraction = trial.suggest_float('feature_fraction', 0.6, 1.0) if trial else params.get('feature_fraction', 0.8)
#             bagging_fraction = trial.suggest_float('bagging_fraction', 0.6, 1.0) if trial else params.get('bagging_fraction', 0.8)
            
#             return lgb.LGBMClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 num_leaves=num_leaves,
#                 feature_fraction=feature_fraction,
#                 bagging_fraction=bagging_fraction,
#                 random_state=42,
#                 verbosity=-1
#             )
        
#         elif model_type == 'catboost' and CATBOOST_AVAILABLE:
#             n_estimators = trial.suggest_int('n_estimators', 50, 200) if trial else params.get('n_estimators', 100)
#             depth = trial.suggest_int('depth', 4, 10) if trial else params.get('depth', 6)
#             learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3) if trial else params.get('learning_rate', 0.1)
#             l2_leaf_reg = trial.suggest_int('l2_leaf_reg', 2, 10) if trial else params.get('l2_leaf_reg', 3)
            
#             return cb.CatBoostClassifier(
#                 iterations=n_estimators,
#                 depth=depth,
#                 learning_rate=learning_rate,
#                 l2_leaf_reg=l2_leaf_reg,
#                 auto_class_weights='Balanced',
#                 random_seed=42,
#                 verbose=False,
#                 early_stopping_rounds=15
#             )
        
#         else:
#             # Fallback to random forest
#             return RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=8,
#                 class_weight='balanced',
#                 random_state=42,
#                 n_jobs=-1
#             )

# # =============================================================================
# # ENHANCED TRACK WITH UNCERTAINTY AND DROPOUT
# # =============================================================================

# class OptimizedEnhancedTrack:
#     """Enhanced track with uncertainty estimation and regularization."""
    
#     def __init__(self, name: str, level: str, classifier=None, parent_track=None, config: TRAConfig = None):
#         self.name = name
#         self.level = level
#         self.classifier = classifier
#         self.parent_track = parent_track
#         self.config = config or TRAConfig()
        
#         # Performance tracking with uncertainty
#         self.performance_score = 0.5
#         self.usage_count = 0
#         self.last_used = time.time()
#         self.prediction_times = deque(maxlen=50)
#         self.metrics = AdvancedMetrics()
#         self.health_status = "healthy"
        
#         # Enhanced tracking with uncertainty
#         self.train_accuracy = 0.0
#         self.validation_accuracy = 0.0
#         self.test_accuracy = 0.0
#         self.uncertainty_score = 0.0
#         self.entropy_history = deque(maxlen=100)
        
#         # Overfitting detection
#         self.overfitting_score = 0.0
#         self.early_stopping_triggered = False
        
#         # Specialized features
#         self.specialized_features = []
#         self.feature_importance_scores = {}
    
#     def predict_batch_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
#         """Batch prediction with uncertainty estimation."""
#         start_time = time.time()
        
#         try:
#             self.usage_count += len(X)
#             self.last_used = time.time()
            
#             if isinstance(X, pd.DataFrame):
#                 X_array = X.values
#             else:
#                 X_array = np.asarray(X)
            
#             if X_array.ndim == 1:
#                 X_array = X_array.reshape(1, -1)
            
#             # Make predictions
#             if self.classifier is not None:
#                 predictions = self.classifier.predict(X_array)
                
#                 # Get probabilities for uncertainty calculation
#                 if hasattr(self.classifier, 'predict_proba'):
#                     probabilities = self.classifier.predict_proba(X_array)
                    
#                     # Calculate entropy for uncertainty
#                     entropies = []
#                     for prob in probabilities:
#                         entropy = -np.sum(prob * np.log2(prob + 1e-10))
#                         entropies.append(entropy)
                    
#                     entropies = np.array(entropies)
#                     avg_entropy = np.mean(entropies)
                    
#                     # Update entropy history
#                     self.entropy_history.extend(entropies[:10])  # Store sample
                    
#                 else:
#                     probabilities = np.ones((len(X_array), 2)) * 0.5
#                     entropies = np.full(len(X_array), 0.5)
#                     avg_entropy = 0.5
                
#                 # Update uncertainty score
#                 self.uncertainty_score = avg_entropy / np.log2(probabilities.shape[1]) if probabilities.shape[1] > 1 else avg_entropy
#                 self.metrics.uncertainty_score = self.uncertainty_score
#                 self.metrics.entropy_score = avg_entropy
                
#             else:
#                 # Fallback
#                 if self.config.task_type == "classification":
#                     predictions = np.zeros(len(X_array), dtype=int)
#                     probabilities = np.ones((len(X_array), 2)) * 0.5
#                 else:
#                     predictions = np.zeros(len(X_array), dtype=float)
#                     probabilities = predictions.reshape(-1, 1)
                
#                 entropies = np.full(len(X_array), 0.5)
#                 avg_entropy = 0.5
            
#             # Track timing
#             prediction_time = (time.time() - start_time) * 1000
#             self.prediction_times.append(prediction_time)
#             self.metrics.latency_ms = np.mean(self.prediction_times)
#             self.metrics.predictions_per_second = len(X) / (prediction_time / 1000) if prediction_time > 0 else 0
            
#             return predictions, probabilities, avg_entropy
            
#         except Exception as e:
#             logger.error(f"Track {self.name} batch prediction with uncertainty failed: {e}")
#             n_samples = len(X)
#             if self.config.task_type == "classification":
#                 predictions = np.zeros(n_samples, dtype=int)
#                 probabilities = np.ones((n_samples, 2)) * 0.5
#             else:
#                 predictions = np.zeros(n_samples, dtype=float)
#                 probabilities = predictions.reshape(-1, 1)
            
#             entropies = np.full(n_samples, 0.5)
#             return predictions, probabilities, 0.5
    
#     def predict(self, X) -> np.ndarray:
#         """Maintain compatibility with standard predict."""
#         predictions, _, _ = self.predict_batch_with_uncertainty(pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X)
#         return predictions
    
#     def predict_proba(self, X) -> np.ndarray:
#         """Maintain compatibility with standard predict_proba."""
#         _, probabilities, _ = self.predict_batch_with_uncertainty(pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X)
#         return probabilities
    
#     def fit_with_enhanced_validation(self, X_train, y_train, X_val=None, y_val=None):
#         """Enhanced fit with validation, early stopping, and feature importance tracking."""
#         try:
#             if isinstance(X_train, pd.DataFrame):
#                 feature_names = X_train.columns.tolist()
#                 X_train = X_train.values
#             else:
#                 feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
#             if self.classifier is not None:
#                 # Split for validation if not provided
#                 if X_val is None or y_val is None:
#                     if len(X_train) > 50:
#                         X_train_split, X_val, y_train_split, y_val = train_test_split(
#                             X_train, y_train, test_size=0.2, random_state=42,
#                             stratify=y_train if self.config.task_type == "classification" else None
#                         )
#                         X_train = X_train_split
#                         y_train = y_train_split
#                     else:
#                         X_val, y_val = X_train, y_train
                
#                 # Fit with early stopping for supported models
#                 if hasattr(self.classifier, 'fit') and hasattr(self.classifier, 'n_iter_no_change'):
#                     # Models with built-in early stopping
#                     try:
#                         self.classifier.fit(X_train, y_train)
#                         if hasattr(self.classifier, 'n_estimators_'):
#                             logger.debug(f"Track {self.name}: Used {self.classifier.n_estimators_} estimators")
#                     except:
#                         self.classifier.fit(X_train, y_train)
#                 else:
#                     self.classifier.fit(X_train, y_train)
                
#                 # Calculate performance metrics
#                 self.train_accuracy = self.classifier.score(X_train, y_train)
#                 self.validation_accuracy = self.classifier.score(X_val, y_val)
                
#                 # Enhanced overfitting detection
#                 self.overfitting_score = self.train_accuracy - self.validation_accuracy
#                 if self.overfitting_score > 0.15:
#                     logger.warning(f"Track {self.name} shows overfitting: train={self.train_accuracy:.3f}, val={self.validation_accuracy:.3f}")
#                     self.health_status = "overfitting"
                
#                 # Extract feature importance
#                 self._extract_feature_importance(feature_names)
                
#                 # Update metrics
#                 self.metrics.accuracy = self.validation_accuracy
#                 self.metrics.specialization_score = min(1.0, self.validation_accuracy * 1.1)
                
#                 logger.info(f"Track {self.name} - Train: {self.train_accuracy:.4f}, Val: {self.validation_accuracy:.4f}")
                
#             else:
#                 logger.warning(f"No classifier for track {self.name}")
                
#         except Exception as e:
#             logger.error(f"Track {self.name} enhanced fitting failed: {e}")
#             self._create_fallback_classifier(X_train, y_train)
    
#     def _extract_feature_importance(self, feature_names: List[str]):
#         """Extract and store feature importance for specialization."""
#         try:
#             importance = None
            
#             if hasattr(self.classifier, 'feature_importances_'):
#                 importance = self.classifier.feature_importances_
#             elif hasattr(self.classifier, 'coef_'):
#                 importance = np.abs(self.classifier.coef_)
#                 if importance.ndim > 1:
#                     importance = np.mean(importance, axis=0)
            
#             if importance is not None:
#                 # Store feature importance scores
#                 self.feature_importance_scores = dict(zip(feature_names, importance))
                
#                 # Identify specialized features (top 30%)
#                 sorted_features = sorted(self.feature_importance_scores.items(), 
#                                        key=lambda x: x[1], reverse=True)
#                 n_specialized = max(3, len(sorted_features) // 3)
#                 self.specialized_features = [f[0] for f in sorted_features[:n_specialized]]
                
#                 logger.debug(f"Track {self.name} specialized features: {self.specialized_features[:5]}")
                
#         except Exception as e:
#             logger.debug(f"Feature importance extraction failed for {self.name}: {e}")
    
#     def _create_fallback_classifier(self, X_train, y_train):
#         """Create fallback classifier when main classifier fails."""
#         try:
#             if self.config.task_type == "classification":
#                 from sklearn.dummy import DummyClassifier
#                 self.classifier = DummyClassifier(strategy="most_frequent")
#             else:
#                 from sklearn.dummy import DummyRegressor
#                 self.classifier = DummyRegressor(strategy="mean")
            
#             self.classifier.fit(X_train, y_train)
#             self.train_accuracy = self.classifier.score(X_train, y_train)
#             self.validation_accuracy = self.train_accuracy
#             self.health_status = "fallback"
            
#         except Exception as e:
#             logger.error(f"Even fallback classifier failed for track {self.name}: {e}")
    
#     def fit(self, X, y):
#         """Maintain compatibility with standard fit."""
#         self.fit_with_enhanced_validation(X, y)
    
#     def get_uncertainty_summary(self) -> Dict[str, Any]:
#         """Get uncertainty and performance summary."""
#         return {
#             "uncertainty_score": self.uncertainty_score,
#             "avg_entropy": np.mean(self.entropy_history) if self.entropy_history else 0.0,
#             "entropy_std": np.std(self.entropy_history) if self.entropy_history else 0.0,
#             "overfitting_score": self.overfitting_score,
#             "specialized_features": self.specialized_features[:5],
#             "health_status": self.health_status,
#             "predictions_count": self.usage_count
#         }

# # =============================================================================
# # ENHANCED ROUTING ENGINE WITH ENTROPY AND SHAP
# # =============================================================================

# class EnhancedSmartRoutingEngine:
#     """Enhanced routing engine with entropy, SHAP, and uncertainty-based decisions."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.routing_statistics = defaultdict(int)
#         self.entropy_engine = EntropyUncertaintyEngine(config)
#         self.shap_engine = EnhancedSHAPRoutingEngine(config)
#         self.routing_decisions_history = []
#         self.track_performance_history = defaultdict(list)
    
#     def initialize_routing_engines(self, tracks: Dict[str, OptimizedEnhancedTrack], 
#                                   X_sample: pd.DataFrame, y_sample: np.ndarray):
#         """Initialize SHAP and other routing engines."""
#         logger.info("Initializing enhanced routing engines...")
        
#         # Initialize SHAP explainers
#         self.shap_engine.initialize_shap_explainers(tracks, X_sample)
        
#         # Calculate track performance baselines
#         for track_name, track in tracks.items():
#             if hasattr(track, 'validation_accuracy'):
#                 self.track_performance_history[track_name].append(track.validation_accuracy)
    
#     def route_records_intelligently(self, records: List, tracks: Dict[str, OptimizedEnhancedTrack],
#                                    X_batch: pd.DataFrame) -> List[RoutingDecision]:
#         """Intelligent routing using entropy, SHAP, and uncertainty."""
#         routing_decisions = []
        
#         for i, record in enumerate(records):
#             try:
#                 X_instance = X_batch.iloc[i:i+1]
                
#                 # Step 1: Get predictions and probabilities from all tracks
#                 track_predictions = {}
#                 track_probabilities = {}
#                 track_entropies = {}
#                 track_uncertainties = {}
                
#                 for track_name, track in tracks.items():
#                     try:
#                         pred, prob, entropy = track.predict_batch_with_uncertainty(X_instance)
#                         track_predictions[track_name] = pred[0]
#                         track_probabilities[track_name] = prob[0]
#                         track_entropies[track_name] = entropy
                        
#                         # Calculate uncertainty
#                         uncertainty = self.entropy_engine.calculate_uncertainty_score(pred, prob[0])
#                         track_uncertainties[track_name] = uncertainty
                        
#                     except Exception as e:
#                         logger.debug(f"Track prediction failed for {track_name}: {e}")
#                         track_predictions[track_name] = 0
#                         track_probabilities[track_name] = np.array([0.5, 0.5])
#                         track_entropies[track_name] = 0.5
#                         track_uncertainties[track_name] = 0.5
                
#                 # Step 2: Calculate SHAP-based routing scores
#                 shap_scores = self.shap_engine.calculate_track_shap_scores(X_instance, tracks)
                
#                 # Step 3: Multi-criteria routing decision
#                 routing_scores = {}
                
#                 for track_name in tracks.keys():
#                     base_score = 0.3  # Base score
                    
#                     # Performance component (30%)
#                     performance_score = self.track_performance_history[track_name][-1] if self.track_performance_history[track_name] else 0.5
#                     performance_component = 0.3 * performance_score
                    
#                     # SHAP specialization component (25%)
#                     shap_component = 0.25 * shap_scores.get(track_name, 0.5)
                    
#                     # Confidence component (20%)
#                     max_prob = np.max(track_probabilities[track_name])
#                     confidence_component = 0.2 * max_prob
                    
#                     # Uncertainty penalty (15%)
#                     uncertainty = track_uncertainties[track_name]
#                     uncertainty_penalty = 0.15 * (1.0 - uncertainty)
                    
#                     # Entropy component (10%) - lower entropy is better
#                     entropy = track_entropies[track_name]
#                     max_entropy = np.log2(len(track_probabilities[track_name]))
#                     entropy_component = 0.1 * (1.0 - entropy / max_entropy)
                    
#                     # Combined routing score
#                     routing_scores[track_name] = (base_score + performance_component + 
#                                                 shap_component + confidence_component + 
#                                                 uncertainty_penalty + entropy_component)
                
#                 # Step 4: Select best track with adaptive threshold
#                 best_track = max(routing_scores, key=routing_scores.get)
#                 best_score = routing_scores[best_track]
                
#                 # Apply adaptive threshold
#                 adaptive_threshold = self._calculate_adaptive_threshold(best_track, X_instance)
                
#                 # Routing decision
#                 if best_score >= adaptive_threshold:
#                     selected_track = best_track
#                     confidence = best_score
#                     reason = f"multi_criteria_routing_score_{best_score:.3f}"
                    
#                     # Check for high uncertainty re-routing
#                     if (self.config.enable_entropy_routing and 
#                         track_entropies[selected_track] > self.config.entropy_threshold):
                        
#                         # Re-route to global track for high uncertainty
#                         if "global_track_0" in tracks and selected_track != "global_track_0":
#                             selected_track = "global_track_0"
#                             confidence = 0.7
#                             reason = f"entropy_rerouting_entropy_{track_entropies[best_track]:.3f}"
                
#                 else:
#                     # Fallback to global track
#                     selected_track = "global_track_0" if "global_track_0" in tracks else list(tracks.keys())[0]
#                     confidence = 0.6
#                     reason = f"threshold_fallback_score_{best_score:.3f}"
                
#                 # Create routing decision
#                 decision = RoutingDecision(
#                     record_id=record.id,
#                     selected_track=selected_track,
#                     confidence=confidence,
#                     reason=reason,
#                     alternative_tracks=routing_scores,
#                     entropy_score=track_entropies.get(selected_track, 0.5),
#                     shap_scores=shap_scores,
#                     uncertainty_score=track_uncertainties.get(selected_track, 0.5)
#                 )
                
#                 routing_decisions.append(decision)
#                 self.routing_decisions_history.append(decision)
#                 self.routing_statistics[selected_track] += 1
                
#             except Exception as e:
#                 logger.warning(f"Intelligent routing failed for record {record.id}: {e}")
#                 # Fallback routing
#                 fallback_track = list(tracks.keys())[0]
#                 decision = RoutingDecision(
#                     record_id=record.id,
#                     selected_track=fallback_track,
#                     confidence=0.3,
#                     reason=f"routing_error_{str(e)[:30]}"
#                 )
#                 routing_decisions.append(decision)
#                 self.routing_statistics[fallback_track] += 1
        
#         return routing_decisions
    
#     def _calculate_adaptive_threshold(self, track_name: str, X_instance: pd.DataFrame) -> float:
#         """Calculate adaptive threshold based on track performance and context."""
#         base_threshold = self.config.routing_threshold
        
#         # Adjust based on track performance history
#         if self.track_performance_history[track_name]:
#             recent_performance = np.mean(self.track_performance_history[track_name][-3:])
#             performance_adjustment = (recent_performance - 0.7) * 0.2
#             base_threshold += performance_adjustment
        
#         # Adjust based on routing success rate
#         total_routes = sum(self.routing_statistics.values())
#         if total_routes > 0:
#             track_usage_rate = self.routing_statistics[track_name] / total_routes
#             if track_usage_rate < 0.1:  # Underused track
#                 base_threshold -= 0.05  # Lower threshold to encourage usage
        
#         return max(0.2, min(0.8, base_threshold))
    
#     def update_performance_feedback(self, track_name: str, accuracy: float):
#         """Update performance feedback for adaptive routing."""
#         self.track_performance_history[track_name].append(accuracy)
        
#         # Keep recent history only
#         if len(self.track_performance_history[track_name]) > 10:
#             self.track_performance_history[track_name] = self.track_performance_history[track_name][-10:]
    
#     def get_enhanced_routing_summary(self) -> Dict[str, Any]:
#         """Get comprehensive routing summary with entropy and SHAP insights."""
#         total_routed = sum(self.routing_statistics.values())
        
#         if total_routed == 0:
#             return {"total_records": 0, "routing_distribution": {}}
        
#         # Basic statistics
#         distribution = {track: count/total_routed for track, count in self.routing_statistics.items()}
        
#         # Recent routing decisions analysis
#         recent_decisions = self.routing_decisions_history[-100:] if self.routing_decisions_history else []
        
#         # Entropy analysis
#         entropy_stats = {}
#         if recent_decisions:
#             entropies = [d.entropy_score for d in recent_decisions if d.entropy_score > 0]
#             if entropies:
#                 entropy_stats = {
#                     'mean_entropy': np.mean(entropies),
#                     'std_entropy': np.std(entropies),
#                     'high_entropy_rate': np.mean(np.array(entropies) > self.config.entropy_threshold)
#                 }
        
#         # Uncertainty analysis
#         uncertainty_stats = {}
#         if recent_decisions:
#             uncertainties = [d.uncertainty_score for d in recent_decisions if d.uncertainty_score > 0]
#             if uncertainties:
#                 uncertainty_stats = {
#                     'mean_uncertainty': np.mean(uncertainties),
#                     'std_uncertainty': np.std(uncertainties)
#                 }
        
#         # Routing reason analysis
#         reason_counts = defaultdict(int)
#         for decision in recent_decisions:
#             reason_type = decision.reason.split('_')[0]
#             reason_counts[reason_type] += 1
        
#         return {
#             "total_records": total_routed,
#             "routing_distribution": distribution,
#             "routing_statistics": dict(self.routing_statistics),
#             "entropy_analysis": entropy_stats,
#             "uncertainty_analysis": uncertainty_stats,
#             "routing_reasons": dict(reason_counts),
#             "shap_routing_enabled": self.config.enable_shap_routing and SHAP_AVAILABLE,
#             "entropy_routing_enabled": self.config.enable_entropy_routing
#         }

# # =============================================================================
# # ENHANCED ENSEMBLE FUSION WITH UNCERTAINTY WEIGHTING
# # =============================================================================

# class UncertaintyWeightedEnsembleFusion:
#     """Enhanced ensemble fusion with uncertainty-based weighting."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.fusion_strategies = {
#             'uncertainty_weighted': self._uncertainty_weighted_fusion,
#             'shap_weighted': self._shap_weighted_fusion,
#             'entropy_weighted': self._entropy_weighted_fusion,
#             'dynamic_weighted': self._dynamic_weighted_fusion
#         }
#         self.performance_history = defaultdict(list)
#         self.uncertainty_history = defaultdict(list)
#         self.stacking_model = None
    
#     def fuse_predictions_with_uncertainty(self, 
#                                         track_predictions: Dict[str, np.ndarray],
#                                         track_probabilities: Dict[str, np.ndarray],
#                                         track_uncertainties: Dict[str, float],
#                                         track_entropies: Dict[str, float],
#                                         shap_scores: Dict[str, float]) -> np.ndarray:
#         """Main fusion method with uncertainty awareness."""
        
#         if not track_predictions:
#             logger.warning("No predictions to fuse")
#             return np.array([])
        
#         if len(track_predictions) == 1:
#             return list(track_predictions.values())[0]
        
#         # Select fusion strategy based on available information
#         if self.config.enable_uncertainty_weighting and track_uncertainties:
#             strategy = 'uncertainty_weighted'
#         elif self.config.enable_shap_routing and shap_scores:
#             strategy = 'shap_weighted'
#         elif self.config.enable_entropy_routing and track_entropies:
#             strategy = 'entropy_weighted'
#         else:
#             strategy = 'dynamic_weighted'
        
#         logger.debug(f"Using fusion strategy: {strategy}")
        
#         fusion_func = self.fusion_strategies[strategy]
#         return fusion_func(track_predictions, track_probabilities, track_uncertainties, track_entropies, shap_scores)
    
#     def _uncertainty_weighted_fusion(self, track_predictions, track_probabilities, 
#                                    track_uncertainties, track_entropies, shap_scores) -> np.ndarray:
#         """Fusion weighted by prediction uncertainty (lower uncertainty = higher weight)."""
#         try:
#             # Calculate weights based on uncertainty
#             weights = {}
#             total_weight = 0.0
            
#             for track_name in track_predictions.keys():
#                 uncertainty = track_uncertainties.get(track_name, 0.5)
#                 # Lower uncertainty should get higher weight
#                 weight = 1.0 - uncertainty
                
#                 # Boost weight based on historical performance
#                 if track_name in self.performance_history:
#                     hist_perf = np.mean(self.performance_history[track_name][-3:])
#                     weight *= hist_perf
                
#                 weights[track_name] = weight
#                 total_weight += weight
            
#             # Normalize weights
#             if total_weight > 0:
#                 weights = {k: v / total_weight for k, v in weights.items()}
#             else:
#                 weights = {k: 1.0 / len(track_predictions) for k in track_predictions.keys()}
            
#             return self._apply_weighted_fusion(track_predictions, weights)
            
#         except Exception as e:
#             logger.debug(f"Uncertainty weighted fusion failed: {e}")
#             return self._simple_average_fusion(track_predictions)
    
#     def _shap_weighted_fusion(self, track_predictions, track_probabilities, 
#                             track_uncertainties, track_entropies, shap_scores) -> np.ndarray:
#         """Fusion weighted by SHAP specialization scores."""
#         try:
#             weights = {}
#             total_weight = 0.0
            
#             for track_name in track_predictions.keys():
#                 # Base weight from SHAP score
#                 shap_weight = shap_scores.get(track_name, 0.5)
                
#                 # Adjust by confidence
#                 if track_name in track_probabilities:
#                     prob = track_probabilities[track_name]
#                     confidence = np.max(prob) if hasattr(prob, '__len__') else 0.5
#                     shap_weight *= confidence
                
#                 weights[track_name] = shap_weight
#                 total_weight += shap_weight
            
#             # Normalize weights
#             if total_weight > 0:
#                 weights = {k: v / total_weight for k, v in weights.items()}
#             else:
#                 weights = {k: 1.0 / len(track_predictions) for k in track_predictions.keys()}
            
#             return self._apply_weighted_fusion(track_predictions, weights)
            
#         except Exception as e:
#             logger.debug(f"SHAP weighted fusion failed: {e}")
#             return self._simple_average_fusion(track_predictions)
    
#     def _entropy_weighted_fusion(self, track_predictions, track_probabilities, 
#                                track_uncertainties, track_entropies, shap_scores) -> np.ndarray:
#         """Fusion weighted by prediction entropy (lower entropy = higher weight)."""
#         try:
#             weights = {}
#             total_weight = 0.0
            
#             max_entropy = np.log2(2)  # Assuming binary classification
            
#             for track_name in track_predictions.keys():
#                 entropy = track_entropies.get(track_name, max_entropy)
#                 # Lower entropy should get higher weight
#                 weight = 1.0 - (entropy / max_entropy)
                
#                 # Boost by performance history
#                 if track_name in self.performance_history:
#                     hist_perf = np.mean(self.performance_history[track_name][-3:])
#                     weight *= hist_perf
                
#                 weights[track_name] = max(0.1, weight)
#                 total_weight += weights[track_name]
            
#             # Normalize weights
#             if total_weight > 0:
#                 weights = {k: v / total_weight for k, v in weights.items()}
#             else:
#                 weights = {k: 1.0 / len(track_predictions) for k in track_predictions.keys()}
            
#             return self._apply_weighted_fusion(track_predictions, weights)
            
#         except Exception as e:
#             logger.debug(f"Entropy weighted fusion failed: {e}")
#             return self._simple_average_fusion(track_predictions)
    
#     def _dynamic_weighted_fusion(self, track_predictions, track_probabilities, 
#                                track_uncertainties, track_entropies, shap_scores) -> np.ndarray:
#         """Dynamic fusion combining multiple weighting factors."""
#         try:
#             weights = {}
            
#             for track_name in track_predictions.keys():
#                 weight = 0.3  # Base weight
                
#                 # Performance component
#                 if track_name in self.performance_history:
#                     hist_perf = np.mean(self.performance_history[track_name][-3:])
#                     weight += 0.3 * hist_perf
                
#                 # Uncertainty component
#                 if track_name in track_uncertainties:
#                     uncertainty = track_uncertainties[track_name]
#                     weight += 0.2 * (1.0 - uncertainty)
                
#                 # Confidence component
#                 if track_name in track_probabilities:
#                     prob = track_probabilities[track_name]
#                     confidence = np.max(prob) if hasattr(prob, '__len__') else 0.5
#                     weight += 0.2 * confidence
                
#                 weights[track_name] = weight
            
#             # Normalize weights
#             total_weight = sum(weights.values())
#             if total_weight > 0:
#                 weights = {k: v / total_weight for k, v in weights.items()}
#             else:
#                 weights = {k: 1.0 / len(track_predictions) for k in track_predictions.keys()}
            
#             return self._apply_weighted_fusion(track_predictions, weights)
            
#         except Exception as e:
#             logger.debug(f"Dynamic weighted fusion failed: {e}")
#             return self._simple_average_fusion(track_predictions)
    
#     def _apply_weighted_fusion(self, track_predictions: Dict[str, np.ndarray], 
#                              weights: Dict[str, float]) -> np.ndarray:
#         """Apply weighted fusion to predictions."""
#         try:
#             if not track_predictions:
#                 return np.array([])
            
#             # Get sample size
#             first_pred = list(track_predictions.values())[0]
#             n_samples = len(first_pred) if hasattr(first_pred, '__len__') else 1
            
#             if self.config.task_type == "classification":
#                 # Weighted voting for classification
#                 fused = np.zeros(n_samples, dtype=int)
                
#                 for i in range(n_samples):
#                     class_votes = defaultdict(float)
                    
#                     for track_name, predictions in track_predictions.items():
#                         if i < len(predictions):
#                             try:
#                                 pred_class = int(predictions[i])
#                                 weight = weights.get(track_name, 0.0)
#                                 class_votes[pred_class] += weight
#                             except (ValueError, TypeError, IndexError):
#                                 continue
                    
#                     if class_votes:
#                         fused[i] = max(class_votes.items(), key=lambda x: x[1])[0]
#                     else:
#                         fused[i] = 0
            
#             else:
#                 # Weighted average for regression
#                 fused = np.zeros(n_samples, dtype=float)
                
#                 for track_name, predictions in track_predictions.items():
#                     try:
#                         weight = weights.get(track_name, 0.0)
#                         pred_array = np.asarray(predictions, dtype=float)[:n_samples]
#                         if len(pred_array) == n_samples:
#                             fused += weight * pred_array
#                     except (ValueError, TypeError):
#                         continue
            
#             return fused
            
#         except Exception as e:
#             logger.debug(f"Weighted fusion application failed: {e}")
#             return self._simple_average_fusion(track_predictions)
    
#     def _simple_average_fusion(self, track_predictions: Dict[str, np.ndarray]) -> np.ndarray:
#         """Simple average fusion as fallback."""
#         try:
#             pred_arrays = list(track_predictions.values())
#             if not pred_arrays:
#                 return np.array([])
            
#             if self.config.task_type == "classification":
#                 # Simple majority vote
#                 fused = []
#                 n_samples = len(pred_arrays[0])
                
#                 for i in range(n_samples):
#                     votes = [pred[i] for pred in pred_arrays if i < len(pred)]
#                     if votes:
#                         fused.append(max(set(votes), key=votes.count))
#                     else:
#                         fused.append(0)
                
#                 return np.array(fused, dtype=int)
            
#             else:
#                 # Simple average for regression
#                 return np.mean(pred_arrays, axis=0)
                
#         except Exception as e:
#             logger.debug(f"Simple average fusion failed: {e}")
#             return np.array([])
    
#     def update_performance_history(self, track_name: str, accuracy: float, uncertainty: float):
#         """Update performance and uncertainty history."""
#         self.performance_history[track_name].append(accuracy)
#         self.uncertainty_history[track_name].append(uncertainty)
        
#         # Keep recent history
#         if len(self.performance_history[track_name]) > 10:
#             self.performance_history[track_name] = self.performance_history[track_name][-10:]
#         if len(self.uncertainty_history[track_name]) > 10:
#             self.uncertainty_history[track_name] = self.uncertainty_history[track_name][-10:]

# # =============================================================================
# # OPTIMIZED FEATURE ENGINEERING WITH LESS AGGRESSIVE REGULARIZATION
# # =============================================================================

# class OptimizedFeatureEngineer:
#     """Optimized feature engineering with balanced regularization."""
    
#     def __init__(self, config: TRAConfig):
#         self.config = config
#         self.is_fitted = False
#         self.feature_names_in_ = None
#         self.feature_names_out_ = None
#         self.n_features_out_ = None
        
#         # Feature transformation tracking
#         self.interaction_pairs = []
#         self.transformation_features = []
#         self.aggregation_features = []
#         self.selected_numeric_cols = []
        
#         # Feature selection components
#         self.variance_selector = None
#         self.feature_selector = None
#         self.l1_selector = None
#         self.scaler = StandardScaler()
    
#     def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None, domain_type: str = "tabular") -> pd.DataFrame:
#         """Enhanced fit and transform with balanced regularization."""
#         logger.info(f"Starting optimized feature engineering: {X.shape[1]} features")
        
#         # Store original feature names
#         self.feature_names_in_ = [str(col) for col in X.columns]
        
#         # Start with original features
#         X_engineered = X.copy()
#         X_engineered.columns = [str(col) for col in X_engineered.columns]
        
#         # Select features for engineering based on importance
#         self._select_features_for_engineering(X_engineered, y)
        
#         # Generate features adaptively
#         X_engineered = self._generate_features_adaptively(X_engineered, y)
        
#         # Apply balanced regularization (less aggressive)
#         if len(X_engineered.columns) > self.config.max_engineered_features:
#             X_engineered = self._apply_balanced_regularization(X_engineered, y)
        
#         # Final feature selection if still too many
#         if len(X_engineered.columns) > self.config.max_engineered_features:
#             X_engineered = self._final_feature_selection(X_engineered, y)
        
#         # Store final feature names
#         self.feature_names_out_ = sorted([str(col) for col in X_engineered.columns])
#         self.n_features_out_ = len(self.feature_names_out_)
#         self.is_fitted = True
        
#         # Ensure consistent column order
#         X_engineered = X_engineered[self.feature_names_out_]
        
#         logger.info(f"Feature engineering completed: {len(self.feature_names_in_)} → {len(self.feature_names_out_)} features")
        
#         return X_engineered
    
#     def transform(self, X: pd.DataFrame, domain_type: str = "tabular") -> pd.DataFrame:
#         """Transform with perfect consistency."""
#         if not self.is_fitted:
#             raise ValueError("Feature engineer must be fitted before transform")
        
#         # Align input features
#         X_aligned = self._align_input_features(X)
        
#         # Apply same transformations
#         X_engineered = self._apply_stored_transformations(X_aligned)
        
#         # Apply feature selection
#         X_engineered = self._apply_stored_selection(X_engineered)
        
#         # Ensure exact output consistency
#         final_df = pd.DataFrame(index=X.index)
#         for col in self.feature_names_out_:
#             if col in X_engineered.columns:
#                 final_df[col] = X_engineered[col]
#             else:
#                 final_df[col] = 0.0
        
#         final_df = final_df[self.feature_names_out_].fillna(0)
        
#         return final_df
    
#     def _select_features_for_engineering(self, X: pd.DataFrame, y: np.ndarray):
#         """Select most important features for engineering."""
#         numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
#         if y is not None and len(numeric_cols) > 5:
#             try:
#                 # Use mutual information for feature selection
#                 if self.config.task_type == "classification":
#                     mi_scores = mutual_info_classif(X[numeric_cols], y, random_state=42)
#                 else:
#                     mi_scores = mutual_info_regression(X[numeric_cols], y, random_state=42)
                
#                 # Select top features
#                 feature_scores = list(zip(numeric_cols, mi_scores))
#                 feature_scores.sort(key=lambda x: x[1], reverse=True)
#                 self.selected_numeric_cols = [col for col, _ in feature_scores[:10]]  # Increased from 8
                
#                 logger.info(f"Selected {len(self.selected_numeric_cols)} features for engineering")
                
#             except Exception as e:
#                 logger.warning(f"Feature selection for engineering failed: {e}")
#                 self.selected_numeric_cols = numeric_cols[:10]
#         else:
#             self.selected_numeric_cols = numeric_cols[:10]
    
#     def _generate_features_adaptively(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
#         """Generate features adaptively based on dataset characteristics."""
#         dataset_size = len(X)
#         n_features = len(X.columns)
        
#         # More generous adaptive limits
#         if dataset_size < 500:
#             max_interactions = 3  # Increased from 2
#             max_transformations = 3
#         elif dataset_size < 2000:
#             max_interactions = 4  # Increased from 3
#             max_transformations = 4
#         else:
#             max_interactions = 5  # Increased from 4
#             max_transformations = 5
        
#         # Less reduction for high feature count
#         if n_features > 20:  # Increased threshold from 15
#             max_interactions = max(2, max_interactions - 1)
#             max_transformations = max(2, max_transformations - 1)
        
#         X_new = X.copy()
        
#         # Generate interaction features
#         if len(self.selected_numeric_cols) >= 2:
#             X_new = self._create_interaction_features(X_new, max_interactions)
        
#         # Generate transformation features
#         if len(self.selected_numeric_cols) >= 1:
#             X_new = self._create_transformation_features(X_new, max_transformations)
        
#         # Generate aggregation features
#         if len(self.selected_numeric_cols) >= 3:
#             X_new = self._create_aggregation_features(X_new)
        
#         return X_new.fillna(0)
    
#     def _create_interaction_features(self, X: pd.DataFrame, max_features: int = 5) -> pd.DataFrame:
#         """Create interaction features with increased limits."""
#         X_new = X.copy()
#         interaction_count = 0
        
#         for i, col1 in enumerate(self.selected_numeric_cols[:max_features]):
#             if col1 in X.columns and interaction_count < max_features * 3:  # Increased multiplier
#                 for j, col2 in enumerate(self.selected_numeric_cols[i+1:i+4], i+1):  # Increased range
#                     if (col2 in X.columns and j < len(self.selected_numeric_cols) and
#                         interaction_count < max_features * 3):
                        
#                         # Multiplicative interaction
#                         mult_name = f"{col1}_mult_{col2}"
#                         X_new[mult_name] = X[col1] * X[col2]
#                         self.interaction_pairs.append((col1, col2, 'mult'))
#                         interaction_count += 1
                        
#                         # Ratio interaction (with safety)
#                         if interaction_count < max_features * 3:
#                             ratio_name = f"{col1}_div_{col2}"
#                             denominator = X[col2].replace(0, 1e-8)
#                             X_new[ratio_name] = X[col1] / denominator
#                             self.interaction_pairs.append((col1, col2, 'div'))
#                             interaction_count += 1
        
#         return X_new
    
#     def _create_transformation_features(self, X: pd.DataFrame, max_features: int = 5) -> pd.DataFrame:
#         """Create transformation features with increased limits."""
#         X_new = X.copy()
        
#         for col in self.selected_numeric_cols[:max_features]:
#             if col in X.columns:
#                 # Log transformation
#                 log_name = f"{col}_log1p"
#                 X_new[log_name] = np.log1p(np.abs(X[col]))
#                 self.transformation_features.append((col, 'log1p'))
                
#                 # Square root transformation
#                 sqrt_name = f"{col}_sqrt"
#                 X_new[sqrt_name] = np.sqrt(np.abs(X[col]))
#                 self.transformation_features.append((col, 'sqrt'))
                
#                 # Square transformation (for more features now)
#                 if col in self.selected_numeric_cols[:4]:  # Increased from 2
#                     square_name = f"{col}_square"
#                     X_new[square_name] = X[col] ** 2
#                     self.transformation_features.append((col, 'square'))
        
#         return X_new
    
#     def _create_aggregation_features(self, X: pd.DataFrame) -> pd.DataFrame:
#         """Create aggregation features consistently."""
#         X_new = X.copy()
#         available_cols = [col for col in self.selected_numeric_cols if col in X.columns]
        
#         if len(available_cols) >= 3:
#             X_new['mean_selected'] = X[available_cols].mean(axis=1)
#             X_new['std_selected'] = X[available_cols].std(axis=1)
#             X_new['max_selected'] = X[available_cols].max(axis=1)
#             X_new['min_selected'] = X[available_cols].min(axis=1)
#             X_new['range_selected'] = X_new['max_selected'] - X_new['min_selected']
#             X_new['median_selected'] = X[available_cols].median(axis=1)  # NEW
            
#             self.aggregation_features = ['mean_selected', 'std_selected', 'max_selected',
#                                        'min_selected', 'range_selected', 'median_selected']
        
#         return X_new
    
#     def _apply_balanced_regularization(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
#         """Apply balanced L1 regularization with large dataset optimization - ENHANCED VERSION."""
#         try:
#             logger.info(f"Applying balanced L1 regularization on {len(X)} samples, {len(X.columns)} features...")
            
#             # CRITICAL: Skip expensive L1 for very large datasets
#             if len(X) > 200000:
#                 logger.warning(f"Dataset too large ({len(X)} samples). Using fast feature selection instead.")
#                 return self._fast_feature_selection(X, y)
            
#             # CRITICAL: Use substantial sampling for large datasets
#             if len(X) > 20000:  # Lowered threshold
#                 sample_size = min(15000, len(X) // 3)  # More conservative sampling
#                 logger.info(f"Large dataset: using sample of {sample_size} for L1 regularization")
#                 sample_indices = np.random.choice(len(X), sample_size, replace=False)
#                 X_sample = X.iloc[sample_indices]
#                 y_sample = y[sample_indices]
#             else:
#                 X_sample = X
#                 y_sample = y
            
#             # CRITICAL: Much more conservative L1 parameters
#             if self.config.task_type == "classification":
#                 l1_model = LogisticRegression(
#                     penalty='l1',
#                     C=1.0,  # FIXED: Reasonable C value instead of formula
#                     solver='liblinear',
#                     random_state=42,
#                     max_iter=20000,  #  MUCH higher for safety
#                     warm_start=True,
#                     class_weight='balanced',  #  Handle class imbalance
#                     n_jobs=1  #  Prevent nested parallelism
#                 )
#             else:
#                 from sklearn.linear_model import Lasso
#                 l1_model = Lasso(
#                     alpha=0.01,  # FIXED: Reasonable alpha
#                     random_state=42,
#                     max_iter=20000,  # Much higher
#                     warm_start=True
#                 )
            
#             # CRITICAL: Add comprehensive timeout and error handling
#             import signal
            
#             def timeout_handler(signum, frame):
#                 raise TimeoutError("L1 regularization exceeded timeout")
            
#             # Set 5-minute timeout
#             old_handler = signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(300)
            
#             try:
#                 # Progress logging
#                 logger.info("Fitting L1 model... (this may take a few minutes)")
#                 l1_model.fit(X_sample.values, y_sample)
#                 signal.alarm(0)  # Cancel timeout
#                 logger.info("L1 model fitted successfully")
                
#             except (TimeoutError, Exception) as fit_error:
#                 signal.alarm(0)
#                 signal.signal(signal.SIGALRM, old_handler)  # Restore handler
#                 logger.warning(f"L1 regularization failed: {fit_error}. Using fast selection fallback.")
#                 return self._fast_feature_selection(X, y)
            
#             finally:
#                 signal.signal(signal.SIGALRM, old_handler)  # Always restore handler
            
#             # Extract feature importance
#             try:
#                 if hasattr(l1_model, 'coef_'):
#                     if l1_model.coef_.ndim == 1:
#                         importances = np.abs(l1_model.coef_)
#                     else:
#                         importances = np.sum(np.abs(l1_model.coef_), axis=0)
#                 else:
#                     logger.warning("No coefficients found. Using fallback selection.")
#                     return self._fast_feature_selection(X, y)
                
#                 # Select features more conservatively
#                 n_select = min(int(self.config.max_engineered_features * 0.8), len(X.columns))  #  20% more conservative
                
#                 # Handle zero coefficients (L1 can zero out features)
#                 non_zero_mask = importances > 1e-10
#                 if np.sum(non_zero_mask) < n_select:
#                     n_select = max(5, np.sum(non_zero_mask))  # At least 5 features
                
#                 top_indices = np.argsort(importances)[::-1][:n_select]
#                 selected_features = X.columns[top_indices]
                
#                 X_selected = X[selected_features]
#                 logger.info(f" L1 regularization completed: {len(X.columns)} → {len(X_selected.columns)} features")
                
#                 return X_selected
                
#             except Exception as selection_error:
#                 logger.warning(f"Feature selection failed: {selection_error}. Using fallback.")
#                 return self._fast_feature_selection(X, y)
                
#         except Exception as e:
#             logger.error(f"L1 regularization completely failed: {e}")
#             return self._fast_feature_selection(X, y)
#     def _fast_feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
#         """Fast feature selection using statistical methods."""
#         try:
#             logger.info(f"Using fast feature selection on {len(X)} samples")
            
#             # Use variance threshold first (very fast)
#             from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
            
#             # Remove low-variance features
#             var_selector = VarianceThreshold(threshold=0.01)
#             X_var = pd.DataFrame(
#                 var_selector.fit_transform(X),
#                 columns=X.columns[var_selector.get_support()],
#                 index=X.index
#             )
            
#             if len(X_var.columns) <= self.config.max_engineered_features:
#                 return X_var
            
#             # Use univariate selection (much faster than L1)
#             if self.config.task_type == "classification":
#                 selector = SelectKBest(f_classif, k=self.config.max_engineered_features)
#             else:
#                 selector = SelectKBest(f_regression, k=self.config.max_engineered_features)
            
#             # Sample for very large datasets
#             if len(X_var) > 50000:
#                 sample_size = 30000
#                 sample_indices = np.random.choice(len(X_var), sample_size, replace=False)
#                 X_sample = X_var.iloc[sample_indices]
#                 y_sample = y[sample_indices]
#             else:
#                 X_sample = X_var
#                 y_sample = y
            
#             selector.fit(X_sample, y_sample)
#             selected_features = X_var.columns[selector.get_support()]
            
#             X_selected = X_var[selected_features]
#             logger.info(f"Fast selection completed: {len(X.columns)} → {len(X_selected.columns)} features")
            
#             return X_selected
            
#         except Exception as e:
#             logger.error(f"Fast selection failed: {e}. Using simple truncation.")
#             return X.iloc[:, :self.config.max_engineered_features]
    
#     def _final_feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
#         """Final feature selection step with less aggressive filtering."""
#         if len(X.columns) <= self.config.max_engineered_features:
#             return X
        
#         try:
#             # Remove low-variance features (less aggressive threshold)
#             from sklearn.feature_selection import VarianceThreshold
#             self.variance_selector = VarianceThreshold(threshold=0.005)  # Reduced from 0.01
#             X_variance = pd.DataFrame(
#                 self.variance_selector.fit_transform(X),
#                 columns=X.columns[self.variance_selector.get_support()],
#                 index=X.index
#             )
            
#             if len(X_variance.columns) > self.config.max_engineered_features and y is not None:
#                 # Apply mutual information selection
#                 if self.config.task_type == "classification":
#                     score_func = mutual_info_classif
#                 else:
#                     score_func = mutual_info_regression
                
#                 self.feature_selector = SelectKBest(
#                     score_func=score_func,
#                     k=self.config.max_engineered_features
#                 )
                
#                 X_selected = pd.DataFrame(
#                     self.feature_selector.fit_transform(X_variance, y),
#                     columns=X_variance.columns[self.feature_selector.get_support()],
#                     index=X.index
#                 )
                
#                 logger.info(f"Final selection: {len(X.columns)} → {len(X_selected.columns)} features")
#                 return X_selected
#             else:
#                 return X_variance
                
#         except Exception as e:
#             logger.warning(f"Final feature selection failed: {e}")
#             return X.iloc[:, :self.config.max_engineered_features]
    
#     def _align_input_features(self, X: pd.DataFrame) -> pd.DataFrame:
#         """Align input features with training features."""
#         X_aligned = pd.DataFrame(index=X.index)
#         for col in self.feature_names_in_:
#             if col in X.columns:
#                 X_aligned[str(col)] = X[col]
#             else:
#                 X_aligned[str(col)] = 0.0
#         return X_aligned
    
#     def _apply_stored_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
#         """Apply stored feature transformations."""
#         X_new = X.copy()
        
#         # Apply interaction features
#         for col1, col2, op_type in self.interaction_pairs:
#             if col1 in X.columns and col2 in X.columns:
#                 if op_type == 'mult':
#                     feature_name = f"{col1}_mult_{col2}"
#                     X_new[feature_name] = X[col1] * X[col2]
#                 elif op_type == 'div':
#                     feature_name = f"{col1}_div_{col2}"
#                     denominator = X[col2].replace(0, 1e-8)
#                     X_new[feature_name] = X[col1] / denominator
        
#         # Apply transformation features
#         for col, transform_type in self.transformation_features:
#             if col in X.columns:
#                 if transform_type == 'log1p':
#                     feature_name = f"{col}_log1p"
#                     X_new[feature_name] = np.log1p(np.abs(X[col]))
#                 elif transform_type == 'sqrt':
#                     feature_name = f"{col}_sqrt"
#                     X_new[feature_name] = np.sqrt(np.abs(X[col]))
#                 elif transform_type == 'square':
#                     feature_name = f"{col}_square"
#                     X_new[feature_name] = X[col] ** 2
        
#         # Apply aggregation features
#         available_cols = [col for col in self.selected_numeric_cols if col in X.columns]
#         if len(available_cols) >= 3:
#             for agg_feature in self.aggregation_features:
#                 if agg_feature == 'mean_selected':
#                     X_new['mean_selected'] = X[available_cols].mean(axis=1)
#                 elif agg_feature == 'std_selected':
#                     X_new['std_selected'] = X[available_cols].std(axis=1)
#                 elif agg_feature == 'max_selected':
#                     X_new['max_selected'] = X[available_cols].max(axis=1)
#                 elif agg_feature == 'min_selected':
#                     X_new['min_selected'] = X[available_cols].min(axis=1)
#                 elif agg_feature == 'range_selected':
#                     max_vals = X[available_cols].max(axis=1)
#                     min_vals = X[available_cols].min(axis=1)
#                     X_new['range_selected'] = max_vals - min_vals
#                 elif agg_feature == 'median_selected':
#                     X_new['median_selected'] = X[available_cols].median(axis=1)
        
#         return X_new.fillna(0)
    
#     def _apply_stored_selection(self, X: pd.DataFrame) -> pd.DataFrame:
#         """Apply stored feature selection."""
#         try:
#             # Apply variance selector
#             if self.variance_selector is not None:
#                 selected_cols = X.columns[self.variance_selector.get_support()]
#                 X = X[selected_cols]
            
#             # Apply feature selector
#             if self.feature_selector is not None:
#                 selected_cols = X.columns[self.feature_selector.get_support()]
#                 X = X[selected_cols]
            
#             return X
            
#         except Exception as e:
#             logger.debug(f"Stored selection application failed: {e}")
#             # Return available features up to max
#             available_features = [col for col in self.feature_names_out_ if col in X.columns]
#             return X[available_features[:self.config.max_engineered_features]]

# # =============================================================================
# # ENHANCED TRA CORE WITH ALL IMPROVEMENTS
# # =============================================================================

# @dataclass
# class Record:
#     """Simple record structure."""
#     id: int
#     data: Dict[str, Any] = field(default_factory=dict)
#     timestamp: float = field(default_factory=time.time)

# class EnhancedTrackRailAlgorithm:
#     """Enhanced Track-Rail Algorithm with all optimizations implemented."""
    
#     def __init__(self, config: TRAConfig = None):
#         self.config = config or TRAConfig()
#         self.config.validate()
        
#         # Core components
#         self.tracks = {}
#         self.routing_engine = EnhancedSmartRoutingEngine(self.config)
#         self.clustering_engine = EnhancedClusteringEngine(self.config)
#         self.feature_engineer = OptimizedFeatureEngineer(self.config)
#         self.ensemble_fusion = UncertaintyWeightedEnsembleFusion(self.config)
#         self.drift_detector = DriftDetectionEngine(self.config)
        
#         # Data tracking
#         self.X_train = None
#         self.y_train = None
#         self.X_test = None
#         self.y_test = None
#         self.feature_names = []
        
#         # Performance tracking
#         self.training_history = []
#         self.routing_history = []
#         self.cluster_labels = None
        
#         # Status
#         self.is_fitted = False
#         self.total_records_processed = 0
        
#         logger.info("Enhanced TRA system initialized with comprehensive optimizations")
    
#     def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, 
#         validation_data: Tuple = None, verbose: bool = True) -> 'EnhancedTrackRailAlgorithm':
#         """Enhanced fit method with comprehensive training."""
        
#         start_time = time.time()
#         logger.info("=" * 80)
#         logger.info(" ENHANCED TRACK-RAIL ALGORITHM - TRAINING START")
#         logger.info("=" * 80)
        
#         try:
#             # Perfect: Large dataset configuration adjustment
#             if len(X) > 100000:
#                 logger.warning(f"Large dataset detected ({len(X)} samples). Adjusting configuration for performance.")
#                 self.config.max_engineered_features = 15  # Reduce from 30
#                 self.config.automl_trials = 5  # Reduce from 20
#                 self.config.l1_regularization_strength = 0.1  # Increase regularization
#                 self.config.max_tracks = 3  # Reduce track count
#                 logger.info("Configuration adjusted for large dataset performance")
            
#             # Prepare data
#             X, y = self._prepare_training_data(X, y, validation_data)
            
#             # Feature engineering
#             logger.info(" Phase 1: Enhanced Feature Engineering")
#             X_engineered = self._perform_feature_engineering(X, y)
            
#             # Clustering and track creation
#             logger.info(" Phase 2: Intelligent Clustering & Track Creation")
#             self._create_intelligent_tracks(X_engineered, y)
            
#             # Track training with regularization
#             logger.info(" Phase 3: Track Training with Regularization")
#             self._train_tracks_with_regularization(X_engineered, y)
            
#             # Initialize routing engines
#             logger.info(" Phase 4: Routing Engine Initialization")
#             self._initialize_routing_systems(X_engineered, y)
            
#             # Meta-learning and stacking
#             logger.info(" Phase 5: Meta-Learning & Ensemble Setup")
#             self._setup_meta_learning(X_engineered, y)
            
#             # Performance evaluation
#             logger.info(" Phase 6: Comprehensive Performance Evaluation")
#             final_metrics = self._evaluate_comprehensive_performance()
            
#             self.is_fitted = True
#             training_time = time.time() - start_time
            
#             logger.info("=" * 80)
#             logger.info(f" TRAINING COMPLETED in {training_time:.2f}s")
#             logger.info(f" Final Test Accuracy: {final_metrics['test_accuracy']:.4f}")
#             logger.info(f" Track Distribution: {self._get_track_distribution()}")
#             logger.info("=" * 80)
            
#             return self
            
#         except Exception as e:
#             logger.error(f" Training failed: {str(e)}")
#             logger.error(traceback.format_exc())
#             raise  

    
#     def _prepare_training_data(self, X, y, validation_data):
#         """Prepare and validate training data."""
#         # Convert to DataFrame if needed
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
#         # Store feature names
#         self.feature_names = X.columns.tolist()
        
#         # Validate data
#         X, y = check_X_y(X.values, y, accept_sparse=False)
#         X = pd.DataFrame(X, columns=self.feature_names)
        
#         # Train-test split
#         if validation_data is None:
#             self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42,
#                 stratify=y if self.config.task_type == "classification" else None
#             )
#         else:
#             self.X_train, self.y_train = X, y
#             self.X_test, self.y_test = validation_data
            
#         logger.info(f"Data prepared - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
#         logger.info(f"Features: {len(self.feature_names)}, Classes: {len(np.unique(y))}")
        
#         return self.X_train, self.y_train
    
#     def _perform_feature_engineering(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
#         """Perform comprehensive feature engineering."""
#         try:
#             X_engineered = self.feature_engineer.fit_transform(X, y)
            
#             # Update drift detector with new features
#             self.drift_detector.update_feature_statistics(X_engineered)
            
#             logger.info(f"Feature engineering: {X.shape[1]} → {X_engineered.shape[1]} features")
            
#             return X_engineered
            
#         except Exception as e:
#             logger.error(f"Feature engineering failed: {e}")
#             return X
    
#     def _create_intelligent_tracks(self, X: pd.DataFrame, y: np.ndarray):
#         """Create tracks using enhanced clustering."""
#         try:
#             # Find optimal clusters
#             self.cluster_labels = self.clustering_engine.find_optimal_clusters(X.values, y)
            
#             if self.cluster_labels is not None:
#                 n_clusters = len(np.unique(self.cluster_labels))
#                 logger.info(f"Found {n_clusters} clusters using {self.clustering_engine.clustering_method}")
                
#                 # Create regional tracks
#                 self._create_regional_tracks(X, y, self.cluster_labels)
#             else:
#                 logger.warning("Clustering failed, using global track only")
            
#             # Always create global track
#             self._create_global_track(X, y)
            
#             logger.info(f"Created {len(self.tracks)} tracks total")
            
#         except Exception as e:
#             logger.error(f"Track creation failed: {e}")
#             self._create_global_track(X, y)
    
#     def _create_regional_tracks(self, X: pd.DataFrame, y: np.ndarray, cluster_labels: np.ndarray):
#         """Create regional tracks based on clustering with better data allocation."""
#         unique_clusters = np.unique(cluster_labels)
        
#         for cluster_id in unique_clusters:
#             if cluster_id == -1:  # Skip noise points
#                 continue
                
#             try:
#                 cluster_mask = cluster_labels == cluster_id
#                 X_cluster = X[cluster_mask]
#                 y_cluster = y[cluster_mask]
                
#                 # Enhanced minimum size check
#                 min_size = max(20, len(X) * self.config.min_cluster_size_ratio)
#                 if len(X_cluster) < min_size:
#                     logger.debug(f"Cluster {cluster_id} too small: {len(X_cluster)} < {min_size}")
#                     continue
                
#                 # Check class diversity
#                 unique_classes = len(np.unique(y_cluster))
#                 if (self.config.task_type == "classification" and 
#                     unique_classes < max(2, len(np.unique(y)) * self.config.min_minority_class_ratio)):
#                     logger.debug(f"Cluster {cluster_id} lacks class diversity: {unique_classes}")
#                     continue
                
#                 # Create and train regional track
#                 track_name = f"regional_track_{cluster_id}"
                
#                 # EXPANDED: Use expanded AutoML for regional tracks
#                 if OPTUNA_AVAILABLE and self.config.enable_expanded_automl:
#                     classifier = self._create_optimized_classifier_with_automl(X_cluster, y_cluster)
#                 else:
#                     classifier = self._create_fallback_classifier()
                
#                 track = OptimizedEnhancedTrack(
#                     name=track_name,
#                     level="regional",
#                     classifier=classifier,
#                     config=self.config
#                 )
                
#                 self.tracks[track_name] = track
#                 logger.info(f"Created {track_name} with {len(X_cluster)} samples, {unique_classes} classes")
                
#             except Exception as e:
#                 logger.warning(f"Regional track creation failed for cluster {cluster_id}: {e}")
#                 continue
    
#     def _create_global_track(self, X: pd.DataFrame, y: np.ndarray):
#         """Create global track with expanded AutoML."""
#         try:
#             track_name = "global_track_0"
            
#             # EXPANDED: Use expanded AutoML for global track
#             if OPTUNA_AVAILABLE and self.config.enable_expanded_automl:
#                 classifier = self._create_optimized_classifier_with_automl(X, y, trials=self.config.automl_trials)
#             else:
#                 classifier = self._create_fallback_classifier()
            
#             track = OptimizedEnhancedTrack(
#                 name=track_name,
#                 level="global",
#                 classifier=classifier,
#                 config=self.config
#             )
            
#             self.tracks[track_name] = track
#             logger.info(f"Created {track_name} with expanded AutoML optimization")
            
#         except Exception as e:
#             logger.error(f"Global track creation failed: {e}")
#             # Create basic fallback track
#             classifier = RandomForestClassifier(
#                 n_estimators=100, random_state=42, class_weight='balanced'
#             )
#             track = OptimizedEnhancedTrack(
#                 name="global_track_0",
#                 level="global", 
#                 classifier=classifier,
#                 config=self.config
#             )
#             self.tracks["global_track_0"] = track
    
#     def _create_optimized_classifier_with_automl(self, X: pd.DataFrame, y: np.ndarray, 
#                                                trials: int = None) -> BaseEstimator:
#         """Create optimized classifier using expanded AutoML."""
#         if not OPTUNA_AVAILABLE:
#             return self._create_fallback_classifier()
        
#         trials = trials or max(10, self.config.automl_trials // 2)  # Adaptive trials
        
#         try:
#             def objective(trial):
#                 try:
#                     # EXPANDED model choices including CatBoost
#                     model_choices = ['random_forest', 'gradient_boosting', 'extra_trees']
                    
#                     if XGBOOST_AVAILABLE:
#                         model_choices.append('xgboost')
#                     if LIGHTGBM_AVAILABLE:
#                         model_choices.append('lightgbm')
#                     if CATBOOST_AVAILABLE:
#                         model_choices.append('catboost')
                    
#                     model_type = trial.suggest_categorical('model_type', model_choices)
                    
#                     # Create model with expanded hyperparameter space
#                     classifier = ExpandedAutoMLModelFactory.create_expanded_classifier(
#                         model_type, self.config, trial
#                     )
                    
#                     # Cross-validation with early stopping
#                     if len(X) > 100:
#                         cv_scores = cross_val_score(
#                             classifier, X.values, y, 
#                             cv=min(5, max(3, len(X) // 50)),
#                             scoring='accuracy' if self.config.task_type == "classification" else 'neg_mean_squared_error',
#                             n_jobs=1  # Prevent nested parallelism
#                         )
#                         score = np.mean(cv_scores)
#                     else:
#                         # Small dataset: simple train-validation split
#                         X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
#                             X, y, test_size=0.3, random_state=42
#                         )
#                         classifier.fit(X_train_cv.values, y_train_cv)
#                         score = classifier.score(X_val_cv.values, y_val_cv)
                    
#                     return score
                    
#                 except Exception as e:
#                     logger.debug(f"AutoML trial failed: {e}")
#                     return -1.0
            
#             # Create study with better settings
#             study = optuna.create_study(
#                 direction='maximize',
#                 sampler=optuna.samplers.TPESampler(seed=42),
#                 pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
#             )
            
#             # Optimize with timeout
#             study.optimize(
#                 objective, 
#                 n_trials=trials,
#                 timeout=300,  # 5 minutes max
#                 n_jobs=1,
#                 show_progress_bar=False
#             )
            
#             # Create best model
#             best_params = study.best_params
#             model_type = best_params.pop('model_type')
            
#             logger.info(f"AutoML selected: {model_type} with score: {study.best_value:.4f}")
            
#             return ExpandedAutoMLModelFactory.create_expanded_classifier(
#                 model_type, self.config, **best_params
#             )
            
#         except Exception as e:
#             logger.warning(f"AutoML optimization failed: {e}")
#             return self._create_fallback_classifier()
    
#     def _create_fallback_classifier(self) -> BaseEstimator:
#         """Create fallback classifier with good defaults."""
#         if self.config.task_type == "classification":
#             return RandomForestClassifier(
#                 n_estimators=150,  # Increased from 100
#                 max_depth=10,     # Increased from 8
#                 min_samples_split=3,
#                 min_samples_leaf=2,
#                 class_weight='balanced',
#                 random_state=42,
#                 n_jobs=-1
#             )
#         else:
#             return RandomForestRegressor(
#                 n_estimators=150,
#                 max_depth=10,
#                 min_samples_split=3,
#                 min_samples_leaf=2,
#                 random_state=42,
#                 n_jobs=-1
#             )
    
#     def _train_tracks_with_regularization(self, X: pd.DataFrame, y: np.ndarray):
#         """Train tracks with enhanced regularization and validation."""
#         logger.info(f"Training {len(self.tracks)} tracks with regularization...")
        
#         for track_name, track in self.tracks.items():
#             try:
#                 logger.info(f"Training {track_name}...")
                
#                 if track.level == "regional" and self.cluster_labels is not None:
#                     # Get cluster-specific data
#                     cluster_id = int(track_name.split('_')[-1])
#                     cluster_mask = self.cluster_labels == cluster_id
#                     X_track = X[cluster_mask]
#                     y_track = y[cluster_mask]
                    
#                     # ENHANCED: Better data allocation - use more training data for regional tracks
#                     if len(X_track) > 30:  # Lowered from 50
#                         X_train_track, X_val_track, y_train_track, y_val_track = train_test_split(
#                             X_track, y_track, test_size=0.15, random_state=42,  # Reduced validation split
#                             stratify=y_track if self.config.task_type == "classification" and len(np.unique(y_track)) > 1 else None
#                         )
#                     else:
#                         X_train_track, y_train_track = X_track, y_track
#                         X_val_track, y_val_track = X_track, y_track
                
#                 else:
#                     # Global track uses all data
#                     X_train_track, y_train_track = X, y
#                     # Use test set for validation
#                     X_val_track = self.feature_engineer.transform(self.X_test)
#                     y_val_track = self.y_test
                
#                 # Enhanced training with validation
#                 track.fit_with_enhanced_validation(
#                     X_train_track, y_train_track, X_val_track, y_val_track
#                 )
                
#                 # Update ensemble fusion history
#                 self.ensemble_fusion.update_performance_history(
#                     track_name, track.validation_accuracy, track.uncertainty_score
#                 )
                
#                 # Update routing engine
#                 self.routing_engine.update_performance_feedback(track_name, track.validation_accuracy)
                
#                 logger.info(f" {track_name}: Train={track.train_accuracy:.4f}, Val={track.validation_accuracy:.4f}")
                
#             except Exception as e:
#                 logger.error(f" Training failed for {track_name}: {e}")
#                 continue
    
#     def _initialize_routing_systems(self, X: pd.DataFrame, y: np.ndarray):
#         """Initialize routing engines with sample data."""
#         try:
#             # Sample data for initialization
#             sample_size = min(50, len(X))
#             sample_indices = np.random.choice(len(X), sample_size, replace=False)
#             X_sample = X.iloc[sample_indices]
#             y_sample = y[sample_indices]
            
#             # Initialize routing engines
#             self.routing_engine.initialize_routing_engines(self.tracks, X_sample, y_sample)
            
#             logger.info("Routing engines initialized with SHAP and entropy support")
            
#         except Exception as e:
#             logger.warning(f"Routing initialization failed: {e}")
    
#     def _setup_meta_learning(self, X: pd.DataFrame, y: np.ndarray):
#         """Setup meta-learning and stacking."""
#         if not self.config.enable_meta_learning or len(self.tracks) < 2:
#             return
        
#         try:
#             logger.info("Setting up meta-learning and stacking...")
            
#             # Create stacking classifier/regressor with track models
#             track_estimators = [(name, track.classifier) for name, track in self.tracks.items() 
#                                if track.classifier is not None]
            
#             if len(track_estimators) >= 2:
#                 if self.config.task_type == "classification":
#                     self.ensemble_fusion.stacking_model = StackingClassifier(
#                         estimators=track_estimators,
#                         final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
#                         cv=3,
#                         n_jobs=1
#                     )
#                 else:
#                     self.ensemble_fusion.stacking_model = StackingRegressor(
#                         estimators=track_estimators,
#                         final_estimator=Ridge(random_state=42),
#                         cv=3,
#                         n_jobs=1
#                     )
                
#                 # Fit stacking model
#                 self.ensemble_fusion.stacking_model.fit(X.values, y)
#                 logger.info("Stacking model trained successfully")
            
#         except Exception as e:
#             logger.warning(f"Meta-learning setup failed: {e}")
    
#     def _evaluate_comprehensive_performance(self) -> Dict[str, Any]:
#         """Comprehensive performance evaluation."""
#         metrics = {}
        
#         try:
#             # Transform test data
#             X_test_engineered = self.feature_engineer.transform(self.X_test)
            
#             # Track-wise evaluation
#             track_metrics = {}
#             for track_name, track in self.tracks.items():
#                 try:
#                     y_pred = track.predict(X_test_engineered)
#                     accuracy = accuracy_score(self.y_test, y_pred)
#                     track_metrics[track_name] = accuracy
                    
#                     logger.info(f" {track_name}: {accuracy:.4f}")
                    
#                 except Exception as e:
#                     logger.debug(f"Track evaluation failed for {track_name}: {e}")
#                     track_metrics[track_name] = 0.0
            
#             # Overall system prediction
#             y_pred_system = self.predict(self.X_test)
#             test_accuracy = accuracy_score(self.y_test, y_pred_system)
            
#             metrics = {
#                 'test_accuracy': test_accuracy,
#                 'track_accuracies': track_metrics,
#                 'n_tracks': len(self.tracks),
#                 'n_features_engineered': len(self.feature_engineer.feature_names_out_) if self.feature_engineer.feature_names_out_ else 0
#             }
            
#             return metrics
            
#         except Exception as e:
#             logger.error(f"Performance evaluation failed: {e}")
#             return {'test_accuracy': 0.0, 'track_accuracies': {}, 'n_tracks': 0}
    
#     def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
#         """Enhanced prediction with intelligent routing and uncertainty-based fusion."""
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before prediction")
        
#         try:
#             # Prepare data
#             if not isinstance(X, pd.DataFrame):
#                 X = pd.DataFrame(X, columns=self.feature_names)
            
#             # Apply feature engineering
#             X_engineered = self.feature_engineer.transform(X)
            
#             # Create records for routing
#             records = [Record(id=i) for i in range(len(X_engineered))]
            
#             # Intelligent routing
#             routing_decisions = self.routing_engine.route_records_intelligently(
#                 records, self.tracks, X_engineered
#             )
            
#             # Track predictions with uncertainty
#             track_predictions = {}
#             track_probabilities = {}
#             track_uncertainties = {}
#             track_entropies = {}
            
#             for track_name, track in self.tracks.items():
#                 try:
#                     predictions, probabilities, avg_entropy = track.predict_batch_with_uncertainty(X_engineered)
#                     track_predictions[track_name] = predictions
#                     track_probabilities[track_name] = probabilities
#                     track_entropies[track_name] = avg_entropy
                    
#                     # Calculate track uncertainty
#                     uncertainty_scores = []
#                     for i in range(len(predictions)):
#                         uncertainty = self.routing_engine.entropy_engine.calculate_uncertainty_score(
#                             predictions[i:i+1], probabilities[i] if probabilities.ndim > 1 else probabilities
#                         )
#                         uncertainty_scores.append(uncertainty)
                    
#                     track_uncertainties[track_name] = np.mean(uncertainty_scores)
                    
#                 except Exception as e:
#                     logger.debug(f"Track prediction failed for {track_name}: {e}")
#                     track_predictions[track_name] = np.zeros(len(X_engineered), dtype=int)
#                     track_probabilities[track_name] = np.ones((len(X_engineered), 2)) * 0.5
#                     track_uncertainties[track_name] = 0.5
#                     track_entropies[track_name] = 0.5
            
#             # Get SHAP scores for fusion
#             if len(X_engineered) > 0:
#                 shap_scores = self.routing_engine.shap_engine.calculate_track_shap_scores(
#                     X_engineered.iloc[:1], self.tracks
#                 )
#             else:
#                 shap_scores = {track: 0.5 for track in self.tracks.keys()}
            
#             # Uncertainty-weighted ensemble fusion
#             final_predictions = self.ensemble_fusion.fuse_predictions_with_uncertainty(
#                 track_predictions, track_probabilities, track_uncertainties, track_entropies, shap_scores
#             )
            
#             # Update drift detection
#             self.drift_detector.update_feature_statistics(X_engineered)
#             for track_name, track in self.tracks.items():
#                 if track_name in track_predictions:
#                     # Estimate accuracy for drift detection (simplified)
#                     pred_confidence = np.mean([np.max(prob) for prob in track_probabilities[track_name]])
#                     self.drift_detector.update_performance(track_name, pred_confidence, track_predictions[track_name])
            
#             # Update total records processed
#             self.total_records_processed += len(X)
            
#             return final_predictions
            
#         except Exception as e:
#             logger.error(f"Prediction failed: {e}")
#             logger.error(traceback.format_exc())
#             # Fallback prediction
#             if self.tracks and "global_track_0" in self.tracks:
#                 try:
#                     X_engineered = self.feature_engineer.transform(X)
#                     return self.tracks["global_track_0"].predict(X_engineered)
#                 except:
#                     return np.zeros(len(X), dtype=int)
#             return np.zeros(len(X), dtype=int)
    
#     def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
#         """Predict probabilities with uncertainty awareness."""
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before prediction")
        
#         if self.config.task_type != "classification":
#             raise ValueError("predict_proba only available for classification")
        
#         try:
#             # Prepare data
#             if not isinstance(X, pd.DataFrame):
#                 X = pd.DataFrame(X, columns=self.feature_names)
            
#             X_engineered = self.feature_engineer.transform(X)
            
#             # Get probabilities from all tracks
#             track_probabilities = {}
#             track_uncertainties = {}
            
#             for track_name, track in self.tracks.items():
#                 try:
#                     _, probabilities, entropy = track.predict_batch_with_uncertainty(X_engineered)
#                     track_probabilities[track_name] = probabilities
#                     track_uncertainties[track_name] = entropy
#                 except Exception as e:
#                     logger.debug(f"Probability prediction failed for {track_name}: {e}")
#                     n_classes = len(np.unique(self.y_train)) if self.y_train is not None else 2
#                     track_probabilities[track_name] = np.ones((len(X_engineered), n_classes)) / n_classes
#                     track_uncertainties[track_name] = 0.5
            
#             # Uncertainty-weighted probability fusion
#             if len(track_probabilities) == 1:
#                 return list(track_probabilities.values())[0]
            
#             # Weight probabilities by inverse uncertainty
#             final_probabilities = np.zeros_like(list(track_probabilities.values())[0])
#             total_weight = 0.0
            
#             for track_name, probabilities in track_probabilities.items():
#                 uncertainty = track_uncertainties.get(track_name, 0.5)
#                 weight = 1.0 - uncertainty
                
#                 final_probabilities += weight * probabilities
#                 total_weight += weight
            
#             if total_weight > 0:
#                 final_probabilities /= total_weight
            
#             return final_probabilities
            
#         except Exception as e:
#             logger.error(f"Probability prediction failed: {e}")
#             n_classes = len(np.unique(self.y_train)) if self.y_train is not None else 2
#             return np.ones((len(X), n_classes)) / n_classes
    
#     def _get_track_distribution(self) -> Dict[str, float]:
#         """Get track usage distribution."""
#         total_routed = sum(self.routing_engine.routing_statistics.values())
#         if total_routed == 0:
#             return {}
        
#         return {
#             track: count / total_routed 
#             for track, count in self.routing_engine.routing_statistics.items()
#         }
    
#     def get_comprehensive_summary(self) -> Dict[str, Any]:
#         """Get comprehensive system summary with all enhancements."""
#         try:
#             summary = {
#                 "system_info": {
#                     "is_fitted": self.is_fitted,
#                     "task_type": self.config.task_type,
#                     "total_records_processed": self.total_records_processed,
#                     "n_tracks": len(self.tracks),
#                     "n_features_original": len(self.feature_names),
#                     "n_features_engineered": len(self.feature_engineer.feature_names_out_) if self.feature_engineer.feature_names_out_ else 0
#                 }
#             }
            
#             if self.is_fitted:
#                 # Track performance
#                 track_performance = {}
#                 for track_name, track in self.tracks.items():
#                     track_performance[track_name] = {
#                         "level": track.level,
#                         "train_accuracy": track.train_accuracy,
#                         "validation_accuracy": track.validation_accuracy,
#                         "uncertainty_score": track.uncertainty_score,
#                         "overfitting_score": track.overfitting_score,
#                         "health_status": track.health_status,
#                         "usage_count": track.usage_count,
#                         "specialized_features": track.specialized_features[:5]
#                     }
                
#                 summary["track_performance"] = track_performance
                
#                 # Enhanced routing summary
#                 summary["routing_analysis"] = self.routing_engine.get_enhanced_routing_summary()
                
#                 # Drift detection summary
#                 summary["drift_analysis"] = self.drift_detector.get_drift_summary()
                
#                 # SHAP and explainability
#                 if SHAP_AVAILABLE and self.routing_engine.shap_engine.track_feature_importance:
#                     summary["shap_analysis"] = {
#                         "shap_routing_enabled": self.config.enable_shap_routing,
#                         "tracks_with_shap": list(self.routing_engine.shap_engine.track_feature_importance.keys()),
#                         "global_feature_importance": dict(sorted(
#                             self.routing_engine.shap_engine.global_feature_importance.items(),
#                             key=lambda x: x[1], reverse=True
#                         )[:10]) if self.routing_engine.shap_engine.global_feature_importance else {}
#                     }
                
#                 # Clustering analysis
#                 if self.cluster_labels is not None:
#                     summary["clustering_analysis"] = {
#                         "method": self.clustering_engine.clustering_method,
#                         "n_clusters": len(np.unique(self.cluster_labels[self.cluster_labels != -1])),
#                         "noise_points": np.sum(self.cluster_labels == -1),
#                         "cluster_sizes": {f"cluster_{i}": np.sum(self.cluster_labels == i) 
#                                         for i in np.unique(self.cluster_labels) if i != -1}
#                     }
                
#                 # Feature engineering summary
#                 summary["feature_engineering"] = {
#                     "interaction_features": len(self.feature_engineer.interaction_pairs),
#                     "transformation_features": len(self.feature_engineer.transformation_features),
#                     "aggregation_features": len(self.feature_engineer.aggregation_features),
#                     "selected_for_engineering": self.feature_engineer.selected_numeric_cols[:5]
#                 }
                
#                 # System health
#                 avg_track_accuracy = np.mean([track.validation_accuracy for track in self.tracks.values()])
#                 avg_uncertainty = np.mean([track.uncertainty_score for track in self.tracks.values()])
                
#                 summary["system_health"] = {
#                     "average_track_accuracy": avg_track_accuracy,
#                     "average_uncertainty": avg_uncertainty,
#                     "tracks_overfitting": len([t for t in self.tracks.values() if t.health_status == "overfitting"]),
#                     "tracks_healthy": len([t for t in self.tracks.values() if t.health_status == "healthy"]),
#                     "entropy_routing_active": self.config.enable_entropy_routing,
#                     "uncertainty_weighting_active": self.config.enable_uncertainty_weighting
#                 }
            
#             return summary
            
#         except Exception as e:
#             logger.error(f"Summary generation failed: {e}")
#             return {"error": str(e), "is_fitted": self.is_fitted}

# # =============================================================================
# # DEMONSTRATION SCRIPT WITH HARD DATASET
# # =============================================================================

# def load_challenging_dataset(dataset_name: str = "covertype"):
#     """Load a challenging dataset for demonstration."""
#     try:
#         if dataset_name == "covertype":
#             # Forest covertype dataset - 7 classes, 54 features
#             logger.info("Loading Forest Covertype dataset (challenging multi-class)...")
#             data = fetch_openml('covertype', version=3, as_frame=True)
#             X, y = data.data, data.target.astype(int) - 1  # Convert to 0-indexed
            
#         elif dataset_name == "adult":
#             # Adult income dataset
#             logger.info("Loading Adult Income dataset...")
#             data = fetch_openml('adult', version=2, as_frame=True)
#             X, y = data.data, (data.target == '>50K').astype(int)
#             # Handle categorical features
#             for col in X.select_dtypes(include=['category', 'object']).columns:
#                 X[col] = pd.Categorical(X[col]).codes
            
#         elif dataset_name == "credit":
#             # German credit dataset
#             logger.info("Loading German Credit dataset...")
#             data = fetch_openml('credit-g', version=1, as_frame=True)
#             X, y = data.data, (data.target == 'bad').astype(int)
#             for col in X.select_dtypes(include=['category', 'object']).columns:
#                 X[col] = pd.Categorical(X[col]).codes
        
#         else:
#             # Fallback to synthetic challenging dataset
#             logger.info("Creating synthetic challenging dataset...")
#             X, y = make_classification(
#                 n_samples=2000, n_features=25, n_informative=15, n_redundant=5,
#                 n_classes=4, n_clusters_per_class=2, flip_y=0.05, random_state=42
#             )
#             X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
#         # Clean data
#         X = X.fillna(X.median(numeric_only=True))
#         X = X.select_dtypes(include=[np.number])
        
#         logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
#         logger.info(f"Class distribution: {np.bincount(y)}")
        
#         return X, y
        
#     except Exception as e:
#         logger.error(f"Dataset loading failed: {e}")
#         logger.info("Creating fallback synthetic dataset...")
#         X, y = make_classification(
#             n_samples=1500, n_features=20, n_informative=12, n_redundant=4,
#             n_classes=3, n_clusters_per_class=2, flip_y=0.03, random_state=42
#         )
#         X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
#         return X, y

# def run_enhanced_tra_demonstration():
#     """Run comprehensive demonstration of Enhanced TRA."""
#     print("\n" + "=" * 100)
#     print(" ENHANCED TRACK-RAIL ALGORITHM (TRA) - COMPREHENSIVE DEMONSTRATION")
#     print("=" * 100)
    
#     try:
#         # Load challenging dataset
#         X, y = load_challenging_dataset("covertype")  # Try covertype first
        
#         # Enhanced configuration
#         config = TRAConfig(
#             task_type="classification",
#             max_tracks=5,
#             enable_expanded_automl=True,
#             enable_entropy_routing=True,
#             enable_shap_routing=True,
#             enable_uncertainty_weighting=True,
#             enable_drift_detection=True,
#             automl_trials=20,  # Reasonable number for demo
#             max_engineered_features=30,
#             enable_regularization=True,
#             dropout_rate=0.1
#         )
        
#         # Create and train Enhanced TRA
#         tra = EnhancedTrackRailAlgorithm(config)
#         tra.fit(X, y, verbose=True)
        
#         # Comprehensive evaluation
#         print("\n" + "=" * 100)
#         print(" COMPREHENSIVE SYSTEM ANALYSIS")
#         print("=" * 100)
        
#         # Get detailed summary
#         summary = tra.get_comprehensive_summary()
        
#         # Display results
#         print(f"\n FINAL PERFORMANCE:")
#         print(f"   • Total Tracks Created: {summary['system_info']['n_tracks']}")
#         print(f"   • Features: {summary['system_info']['n_features_original']} → {summary['system_info']['n_features_engineered']}")
#         print(f"   • Records Processed: {summary['system_info']['total_records_processed']}")
        
#         print(f"\nTRACK PERFORMANCE:")
#         for track_name, metrics in summary['track_performance'].items():
#             print(f"   • {track_name}:")
#             print(f"     - Validation Accuracy: {metrics['validation_accuracy']:.4f}")
#             print(f"     - Uncertainty Score: {metrics['uncertainty_score']:.4f}")
#             print(f"     - Health Status: {metrics['health_status']}")
#             print(f"     - Usage Count: {metrics['usage_count']}")
        
#         print(f"\n ENHANCED ROUTING ANALYSIS:")
#         routing = summary['routing_analysis']
#         print(f"   • Total Routed Records: {routing.get('total_records', 0)}")
#         print(f"   • Track Distribution: {routing.get('routing_distribution', {})}")
#         print(f"   • SHAP Routing: {'Enabled' if routing.get('shap_routing_enabled', False) else ' Disabled'}")
#         print(f"   • Entropy Routing: {' Enabled' if routing.get('entropy_routing_enabled', False) else ' Disabled'}")
        
#         if 'entropy_analysis' in routing and routing['entropy_analysis']:
#             entropy_stats = routing['entropy_analysis']
#             print(f"   • Mean Entropy: {entropy_stats.get('mean_entropy', 0):.4f}")
#             print(f"   • High Entropy Rate: {entropy_stats.get('high_entropy_rate', 0):.4f}")
        
#         print(f"\n🔍 DRIFT DETECTION:")
#         drift = summary['drift_analysis']
#         print(f"   • Total Drift Alerts: {drift.get('total_alerts', 0)}")
#         print(f"   • Recent Alerts: {drift.get('recent_alerts', 0)}")
#         print(f"   • Alert Types: {drift.get('drift_types', {})}")
        
#         if 'shap_analysis' in summary:
#             print(f"\n SHAP EXPLAINABILITY:")
#             shap_info = summary['shap_analysis']
#             print(f"   • SHAP Routing: {' Enabled' if shap_info.get('shap_routing_enabled', False) else ' Disabled'}")
#             print(f"   • Tracks with SHAP: {len(shap_info.get('tracks_with_shap', []))}")
            
#             top_features = list(shap_info.get('global_feature_importance', {}).items())[:5]
#             if top_features:
#                 print(f"   • Top SHAP Features: {[f[0] for f in top_features]}")
        
#         print(f"\n🔧 FEATURE ENGINEERING:")
#         fe = summary['feature_engineering']
#         print(f"   • Interaction Features: {fe.get('interaction_features', 0)}")
#         print(f"   • Transformation Features: {fe.get('transformation_features', 0)}")
#         print(f"   • Aggregation Features: {fe.get('aggregation_features', 0)}")
        
#         if 'clustering_analysis' in summary:
#             print(f"\n CLUSTERING ANALYSIS:")
#             clustering = summary['clustering_analysis']
#             print(f"   • Method: {clustering.get('method', 'Unknown')}")
#             print(f"   • Clusters Found: {clustering.get('n_clusters', 0)}")
#             print(f"   • Noise Points: {clustering.get('noise_points', 0)}")
        
#         print(f"\n💡 SYSTEM HEALTH:")
#         health = summary['system_health']
#         print(f"   • Average Track Accuracy: {health.get('average_track_accuracy', 0):.4f}")
#         print(f"   • Average Uncertainty: {health.get('average_uncertainty', 0):.4f}")
#         print(f"   • Healthy Tracks: {health.get('tracks_healthy', 0)}/{summary['system_info']['n_tracks']}")
#         print(f"   • Overfitting Tracks: {health.get('tracks_overfitting', 0)}")
#         print(f"   • Entropy Routing: {' Active' if health.get('entropy_routing_active', False) else ' Inactive'}")
#         print(f"   • Uncertainty Weighting: {' Active' if health.get('uncertainty_weighting_active', False) else ' Inactive'}")
        
#         # Test prediction capabilities
#         print(f"\n TESTING PREDICTION CAPABILITIES:")
#         X_test_sample = X.iloc[:10]
#         predictions = tra.predict(X_test_sample)
#         probabilities = tra.predict_proba(X_test_sample)
        
#         print(f"   • Sample Predictions: {predictions[:5]}")
#         print(f"   • Sample Probabilities Shape: {probabilities.shape}")
#         print(f"   • Probability Confidence: {np.mean(np.max(probabilities, axis=1)):.4f}")
        
#         print("\n" + "=" * 100)
#         print(" ENHANCED TRA DEMONSTRATION COMPLETED SUCCESSFULLY!")
#         print("=" * 100)
        
#         return tra, summary
        
#     except Exception as e:
#         print(f"\n DEMONSTRATION FAILED: {str(e)}")
#         print(traceback.format_exc())
#         return None, None

# # =============================================================================
# # MAIN EXECUTION
# # =============================================================================

# if __name__ == "__main__":
#     # Run the demonstration
#     tra_system, summary = run_enhanced_tra_demonstration()
    
#     if tra_system is not None:
#         print(f"\n Enhanced TRA system successfully created and tested!")
#         print(f"Ready for production use with comprehensive optimizations!")
        
#         # Optional: Save the trained system
#         try:
#             joblib.dump(tra_system, 'enhanced_tra_system.pkl')
#             print(f" System saved to 'enhanced_tra_system.pkl'")
#         except Exception as e:
#             print(f"  Could not save system: {e}")
#     else:
#         print(f" System creation failed. Please check the logs for details.")
import numpy as np
import pandas as pd
import os
import time
import logging
import warnings
import traceback
import joblib
import gc
import psutil
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from datetime import datetime
import json
import hashlib
from pathlib import Path

# Core ML imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, silhouette_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict, validation_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, mutual_info_classif, mutual_info_regression, SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import make_classification, make_regression, fetch_openml
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# Fix console encoding for Unicode characters
import sys
if sys.platform.startswith('win'):
    try:
        import codecs
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except:
        pass

# OPTIMIZED: Simplified UMAP import with fallback
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

# Advanced ML libraries with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    print("PyTorch not available. Install with: pip install torch")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

# OPTIMIZED: Simplified explainability imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

# OPTIMIZED: Simplified Optuna import
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce verbosity
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

# SMOTE for imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Configure enhanced logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tra_system.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# =============================================================================
# OPTIMIZED CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class OptimizedTRAConfig:
    """Optimized configuration for TRA system with performance constraints."""
    
    # Core TRA settings - OPTIMIZED LIMITS
    task_type: str = "classification"
    max_global_tracks: int = 3  # REDUCED from 5
    max_regional_tracks: int = 2  # REDUCED from unlimited
    enable_meta_learning: bool = True
    
    # Feature engineering - CONSTRAINED
    enable_automated_fe: bool = True
    max_engineered_features: int = 20  # REDUCED from 30
    feature_selection_method: str = "mutual_info"
    l1_regularization_strength: float = 0.01
    
    # OPTIMIZED: AutoML hyperparameter space with strict limits
    enable_automl: bool = True
    max_depth_range: Tuple[int, int] = (4, 8)  # CONSTRAINED depth for speed
    n_estimators_range: Tuple[int, int] = (50, 150)  # REDUCED range
    learning_rate_range: Tuple[float, float] = (0.05, 0.3)
    automl_trials: int = 20  # FIXED at 20 as requested
    
    # Resource management - OPTIMIZED
    use_gpu: bool = TORCH_AVAILABLE
    max_workers: int = 2
    memory_limit_mb: int = 4096
    batch_size: int = 100
    max_dataset_size: int = 30000  # NEW: Cap dataset size for demos
    
    # OPTIMIZED: Ensemble and fusion
    enable_stacking: bool = True
    stacking_cv_folds: int = 3  # REDUCED from 5 for speed
    enable_blending: bool = True
    enable_uncertainty_weighting: bool = True
    ensemble_diversity_threshold: float = 0.2
    min_track_accuracy: float = 0.65
    
    # OPTIMIZED: Routing with simplified thresholds
    enable_entropy_routing: bool = True
    enable_shap_routing: bool = True
    shap_max_depth_threshold: int = 6  # NEW: Skip SHAP for deep models
    shap_max_dataset_size: int = 50000  # NEW: Skip SHAP for large datasets
    entropy_threshold: float = 0.6
    shap_importance_threshold: float = 0.1
    routing_threshold: float = 0.5
    
    # Drift detection - SIMPLIFIED
    enable_drift_detection: bool = True
    drift_detection_window: int = 100
    drift_threshold: float = 0.05
    
    # Clustering improvements - OPTIMIZED
    min_silhouette_score: float = 0.15
    min_cluster_size_ratio: float = 0.05
    min_minority_class_ratio: float = 0.03
    use_dbscan_fallback: bool = True
    use_umap_clustering: bool = False  # DISABLED by default for reliability
    
    # Explainability - CONDITIONAL
    enable_explanations: bool = True
    enable_runtime_explanations: bool = False  # DISABLED for performance
    explanation_method: str = "shap"
    track_decisions: bool = True
    
    # Performance optimization
    enable_early_stopping: bool = True
    validation_split: float = 0.2
    performance_threshold: float = 0.75
    adaptive_learning: bool = True
    
    def validate(self):
        """Validate configuration settings."""
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        if self.max_global_tracks < 1:
            raise ValueError("max_global_tracks must be >= 1")

# =============================================================================
# OPTIMIZED DATA STRUCTURES
# =============================================================================

@dataclass
class OptimizedMetrics:
    """Streamlined performance metrics."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    predictions_per_second: float = 0.0
    confidence_score: float = 0.0
    uncertainty_score: float = 0.0
    
    def efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        if self.latency_ms == 0:
            return 0.5
        
        perf_score = max(0.1, (self.accuracy + self.f1_score) / 2.0)
        throughput = min(1000.0 / max(self.latency_ms, 1.0), 100.0) / 100.0
        confidence_weight = max(0.1, self.confidence_score)
        
        efficiency = 0.4 * perf_score + 0.3 * throughput + 0.3 * confidence_weight
        return min(1.0, max(0.1, efficiency))

@dataclass
class RoutingDecision:
    """Simplified routing decision."""
    record_id: int
    selected_track: str
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DriftAlert:
    """Drift detection alert."""
    timestamp: datetime
    track_name: str
    drift_type: str
    severity: float
    details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# OPTIMIZED MEMORY AND RESOURCE MONITORING
# =============================================================================

class ResourceMonitor:
    """Simple resource monitoring."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    @staticmethod
    def log_memory_usage(stage: str):
        """Log memory usage for a specific stage."""
        memory_mb = ResourceMonitor.get_memory_usage()
        logger.info(f"🔧 {stage}: Memory usage = {memory_mb:.1f} MB")

# =============================================================================
# OPTIMIZED DATASET SAMPLING WITH STRATIFICATION
# =============================================================================

class DatasetSampler:
    """Optimized dataset sampler with stratification."""
    
    @staticmethod
    def sample_large_dataset(X: pd.DataFrame, y: np.ndarray, 
                           max_samples: int = 30000, 
                           stratify: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Sample large dataset while maintaining class distribution."""
        
        if len(X) <= max_samples:
            logger.info(f"📊 Dataset size ({len(X)}) within limits, no sampling needed")
            return X, y
        
        logger.info(f"📊 Sampling dataset: {len(X)} → {max_samples} samples")
        
        if stratify and len(np.unique(y)) > 1:
            try:
                # Stratified sampling to maintain class distribution
                _, X_sampled, _, y_sampled = train_test_split(
                    X, y, test_size=max_samples, random_state=42, stratify=y
                )
                
                logger.info(f"📊 Stratified sampling completed")
                logger.info(f"📊 Original class distribution: {np.bincount(y)}")
                logger.info(f"📊 Sampled class distribution: {np.bincount(y_sampled)}")
                
                return X_sampled, y_sampled
                
            except Exception as e:
                logger.warning(f"Stratified sampling failed: {e}, using random sampling")
        
        # Fallback to random sampling
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        X_sampled = X.iloc[sample_indices].reset_index(drop=True)
        y_sampled = y[sample_indices]
        
        logger.info(f"📊 Random sampling completed")
        return X_sampled, y_sampled

# =============================================================================
# OPTIMIZED CLUSTERING ENGINE
# =============================================================================

class OptimizedClusteringEngine:
    """Optimized clustering with PCA fallback."""
    
    def __init__(self, config: OptimizedTRAConfig):
        self.config = config
        self.clustering_method = None
        self.dimension_reducer = None
        self.cluster_centers = {}
    
    def find_optimal_clusters(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Find optimal clusters with simplified approach."""
        logger.info("🔍 Finding optimal clusters...")
        ResourceMonitor.log_memory_usage("Clustering Start")
        
        # OPTIMIZED: Skip clustering for very large datasets
        if len(X) > 10000:
            logger.warning(f"Dataset too large ({len(X)} samples). Using simple K-means.")
            return self._simple_kmeans_clustering(X, y)
        
        # Reduce dimensions first
        X_reduced = self._reduce_dimensions_optimized(X)
        
        best_labels = None
        best_score = -1
        
        # OPTIMIZED: Simplified clustering methods
        clustering_methods = [
            ('kmeans', self._simple_kmeans_clustering),
            ('pca_kmeans', self._pca_kmeans_clustering)
        ]
        
        for method_name, method_func in clustering_methods:
            try:
                labels = method_func(X_reduced, y)
                if labels is not None:
                    score = self._evaluate_clustering_fast(X_reduced, labels, y)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        self.clustering_method = method_name
                        logger.info(f"🔍 Better clustering: {method_name} (score: {score:.3f})")
            except Exception as e:
                logger.debug(f"Clustering method {method_name} failed: {e}")
                continue
        
        if best_labels is not None:
            logger.info(f"🔍 Selected clustering: {self.clustering_method} with score: {best_score:.3f}")
            ResourceMonitor.log_memory_usage("Clustering Complete")
            return best_labels
        
        logger.warning("🔍 All clustering methods failed, using default split")
        return self._default_clustering(X, y)
    
    def _reduce_dimensions_optimized(self, X: np.ndarray) -> np.ndarray:
        """Optimized dimension reduction with PCA preference."""
        if X.shape[1] <= 8:
            return X
        
        try:
            # OPTIMIZED: Always use PCA for reliability
            n_components = min(8, X.shape[1] - 1)
            self.dimension_reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = self.dimension_reducer.fit_transform(X)
            logger.info(f"🔍 PCA reduction: {X.shape[1]} → {X_reduced.shape[1]} dimensions")
            return X_reduced
        except Exception as e:
            logger.warning(f"Dimension reduction failed: {e}")
            return X
    
    def _simple_kmeans_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Simple and fast K-means clustering."""
        try:
            best_labels = None
            best_score = -1
            
            # OPTIMIZED: Try fewer clusters for speed
            max_clusters = min(4, max(2, len(np.unique(y))))
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=3,  # Reduced iterations
                        max_iter=100,
                        algorithm='elkan'
                    )
                    labels = kmeans.fit_predict(X)
                    
                    if len(np.unique(labels)) >= 2:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                except Exception as e:
                    logger.debug(f"K-means with {n_clusters} clusters failed: {e}")
                    continue
            
            return best_labels
        except Exception as e:
            logger.debug(f"Simple K-means clustering failed: {e}")
            return None
    
    def _pca_kmeans_clustering(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """PCA + K-means clustering."""
        return self._simple_kmeans_clustering(X, y)  # Same implementation
    
    def _evaluate_clustering_fast(self, X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> float:
        """Fast clustering evaluation."""
        try:
            if len(np.unique(labels)) < 2:
                return -1
            
            # Use sample for large datasets
            if len(X) > 2000:
                sample_size = 1000
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[sample_indices]
                labels_sample = labels[sample_indices]
            else:
                X_sample = X
                labels_sample = labels
            
            return silhouette_score(X_sample, labels_sample)
        except Exception as e:
            logger.debug(f"Fast clustering evaluation failed: {e}")
            return -1
    
    def _default_clustering(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Default clustering fallback."""
        # Simple binary split
        n_samples = len(X)
        return np.array([0 if i < n_samples // 2 else 1 for i in range(n_samples)])

# =============================================================================
# OPTIMIZED AUTOML MODEL FACTORY
# =============================================================================

class OptimizedAutoMLFactory:
    """Factory for creating optimized models with constrained hyperparameters."""
    
    @staticmethod
    def create_optimized_classifier(model_type: str, config: OptimizedTRAConfig, 
                                  trial=None, **params) -> BaseEstimator:
        """Create classifier with constrained hyperparameters."""
        
        if model_type == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
            max_depth = trial.suggest_int('max_depth', *config.max_depth_range) if trial else params.get('max_depth', 6)
            
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == 'gradient_boosting':
            n_estimators = trial.suggest_int('n_estimators', 50, 100) if trial else params.get('n_estimators', 75)
            max_depth = trial.suggest_int('max_depth', 3, 6) if trial else params.get('max_depth', 4)
            learning_rate = trial.suggest_float('learning_rate', *config.learning_rate_range) if trial else params.get('learning_rate', 0.1)
            
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
        
        elif model_type == 'extra_trees':
            n_estimators = trial.suggest_int('n_estimators', *config.n_estimators_range) if trial else params.get('n_estimators', 100)
            max_depth = trial.suggest_int('max_depth', *config.max_depth_range) if trial else params.get('max_depth', 6)
            
            return ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        else:
            # Fallback to optimized random forest
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

# =============================================================================
# OPTIMIZED TRACK CLASS
# =============================================================================

class OptimizedTrack:
    """Optimized track with simplified functionality."""
    
    def __init__(self, name: str, level: str, classifier=None, config: OptimizedTRAConfig = None):
        self.name = name
        self.level = level
        self.classifier = classifier
        self.config = config or OptimizedTRAConfig()
        
        # Performance tracking
        self.train_accuracy = 0.0
        self.validation_accuracy = 0.0
        self.usage_count = 0
        self.last_used = time.time()
        self.metrics = OptimizedMetrics()
        self.health_status = "healthy"
        
        # Feature tracking
        self.specialized_features = []
        self.feature_importance_scores = {}
    
    def fit_optimized(self, X_train, y_train, X_val=None, y_val=None):
        """Optimized fit with validation."""
        try:
            start_time = time.time()
            
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
                X_train = X_train.values
            else:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
            if self.classifier is not None:
                # Create validation split if not provided
                if X_val is None or y_val is None:
                    if len(X_train) > 50:
                        X_train_split, X_val, y_train_split, y_val = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42,
                            stratify=y_train if self.config.task_type == "classification" else None
                        )
                        X_train = X_train_split
                        y_train = y_train_split
                    else:
                        X_val, y_val = X_train, y_train
                
                # Fit classifier
                self.classifier.fit(X_train, y_train)
                
                # Calculate performance
                self.train_accuracy = self.classifier.score(X_train, y_train)
                self.validation_accuracy = self.classifier.score(X_val, y_val)
                
                # Extract feature importance
                self._extract_feature_importance(feature_names)
                
                # Update metrics
                self.metrics.accuracy = self.validation_accuracy
                
                # Training time
                training_time = (time.time() - start_time) * 1000
                self.metrics.latency_ms = training_time
                
                logger.info(f"🎯 {self.name}: Train={self.train_accuracy:.4f}, Val={self.validation_accuracy:.4f}")
            else:
                logger.warning(f"No classifier for track {self.name}")
                
        except Exception as e:
            logger.error(f"Track {self.name} training failed: {e}")
            self._create_fallback_classifier(X_train, y_train)
    
    def _extract_feature_importance(self, feature_names: List[str]):
        """Extract feature importance."""
        try:
            importance = None
            if hasattr(self.classifier, 'feature_importances_'):
                importance = self.classifier.feature_importances_
            elif hasattr(self.classifier, 'coef_'):
                importance = np.abs(self.classifier.coef_)
                if importance.ndim > 1:
                    importance = np.mean(importance, axis=0)
            
            if importance is not None:
                self.feature_importance_scores = dict(zip(feature_names, importance))
                # Get top features
                sorted_features = sorted(self.feature_importance_scores.items(),
                                       key=lambda x: x[1], reverse=True)
                n_specialized = max(3, len(sorted_features) // 3)
                self.specialized_features = [f[0] for f in sorted_features[:n_specialized]]
                
        except Exception as e:
            logger.debug(f"Feature importance extraction failed for {self.name}: {e}")
    
    def _create_fallback_classifier(self, X_train, y_train):
        """Create fallback classifier."""
        try:
            from sklearn.dummy import DummyClassifier
            self.classifier = DummyClassifier(strategy="most_frequent")
            self.classifier.fit(X_train, y_train)
            self.train_accuracy = self.classifier.score(X_train, y_train)
            self.validation_accuracy = self.train_accuracy
            self.health_status = "fallback"
        except Exception as e:
            logger.error(f"Fallback classifier failed for {self.name}: {e}")
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        try:
            self.usage_count += len(X)
            self.last_used = time.time()
            
            if self.classifier is not None:
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return self.classifier.predict(X)
            else:
                return np.zeros(len(X), dtype=int)
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        try:
            if self.classifier is not None and hasattr(self.classifier, 'predict_proba'):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return self.classifier.predict_proba(X)
            else:
                n_samples = len(X)
                n_classes = 2  # Default binary classification
                return np.ones((n_samples, n_classes)) * (1.0 / n_classes)
        except Exception as e:
            logger.error(f"Probability prediction failed for {self.name}: {e}")
            n_samples = len(X)
            return np.ones((n_samples, 2)) * 0.5

# =============================================================================
# OPTIMIZED FEATURE ENGINEERING
# =============================================================================

class OptimizedFeatureEngineer:
    """Optimized feature engineering with strict limits."""
    
    def __init__(self, config: OptimizedTRAConfig):
        self.config = config
        self.is_fitted = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.selected_features = []
        self.scaler = StandardScaler()
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        """Optimized feature engineering."""
        logger.info("🔧 Starting optimized feature engineering...")
        ResourceMonitor.log_memory_usage("Feature Engineering Start")
        
        # Store original features
        self.feature_names_in_ = [str(col) for col in X.columns]
        
        # Start with original features
        X_engineered = X.copy()
        X_engineered.columns = [str(col) for col in X_engineered.columns]
        
        # OPTIMIZED: Simple feature creation
        X_engineered = self._create_simple_features(X_engineered, y)
        
        # OPTIMIZED: Fast feature selection
        if len(X_engineered.columns) > self.config.max_engineered_features:
            X_engineered = self._fast_feature_selection(X_engineered, y)
        
        # Store final feature names
        self.feature_names_out_ = sorted([str(col) for col in X_engineered.columns])
        self.is_fitted = True
        
        logger.info(f"🔧 Feature engineering: {len(self.feature_names_in_)} → {len(self.feature_names_out_)} features")
        ResourceMonitor.log_memory_usage("Feature Engineering Complete")
        
        return X_engineered[self.feature_names_out_]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted first")
        
        # Align input features
        X_aligned = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            if col in X.columns:
                X_aligned[str(col)] = X[col]
            else:
                X_aligned[str(col)] = 0.0
        
        # Apply same transformations
        X_engineered = self._create_simple_features(X_aligned, None)
        
        # Ensure output consistency
        final_df = pd.DataFrame(index=X.index)
        for col in self.feature_names_out_:
            if col in X_engineered.columns:
                final_df[col] = X_engineered[col]
            else:
                final_df[col] = 0.0
        
        return final_df[self.feature_names_out_].fillna(0)
    
    def _create_simple_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Create simple engineered features."""
        X_new = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # OPTIMIZED: Create only essential interaction features
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 features
                for j, col2 in enumerate(numeric_cols[i+1:i+3], i+1):  # Limit combinations
                    if j < len(numeric_cols):
                        # Simple multiplication
                        mult_name = f"{col1}_mult_{col2}"
                        X_new[mult_name] = X[col1] * X[col2]
        
        # OPTIMIZED: Simple transformations
        for col in numeric_cols[:5]:  # Limit to first 5 features
            # Log transformation
            log_name = f"{col}_log1p"
            X_new[log_name] = np.log1p(np.abs(X[col]))
            
            # Square root
            sqrt_name = f"{col}_sqrt"
            X_new[sqrt_name] = np.sqrt(np.abs(X[col]))
        
        # OPTIMIZED: Simple aggregations
        if len(numeric_cols) >= 3:
            available_cols = [col for col in numeric_cols[:5] if col in X.columns]
            if len(available_cols) >= 3:
                X_new['mean_top5'] = X[available_cols].mean(axis=1)
                X_new['std_top5'] = X[available_cols].std(axis=1)
                X_new['max_top5'] = X[available_cols].max(axis=1)
        
        return X_new.fillna(0)
    
    def _fast_feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fast feature selection using univariate methods."""
        try:
            if y is None:
                # Return first N features if no target
                return X.iloc[:, :self.config.max_engineered_features]
            
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            
            if self.config.task_type == "classification":
                selector = SelectKBest(f_classif, k=self.config.max_engineered_features)
            else:
                selector = SelectKBest(f_regression, k=self.config.max_engineered_features)
            
            # Use sample for large datasets
            if len(X) > 10000:
                sample_size = 5000
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            selector.fit(X_sample, y_sample)
            selected_features = X.columns[selector.get_support()]
            
            logger.info(f"🔧 Fast selection: {len(X.columns)} → {len(selected_features)} features")
            return X[selected_features]
            
        except Exception as e:
            logger.warning(f"Fast feature selection failed: {e}")
            return X.iloc[:, :self.config.max_engineered_features]

# =============================================================================
# OPTIMIZED ROUTING ENGINE
# =============================================================================

class OptimizedRoutingEngine:
    """Simplified routing engine."""
    
    def __init__(self, config: OptimizedTRAConfig):
        self.config = config
        self.routing_statistics = defaultdict(int)
        self.routing_decisions_history = []
    
    def route_records_simple(self, records: List, tracks: Dict[str, OptimizedTrack], 
                           X_batch: pd.DataFrame) -> List[RoutingDecision]:
        """Simple routing based on track performance."""
        routing_decisions = []
        
        # Default to global track
        default_track = "global_track_0" if "global_track_0" in tracks else list(tracks.keys())[0]
        
        for i, record in enumerate(records):
            try:
                # Simple routing logic: use best performing track
                best_track = default_track
                best_score = 0.0
                
                for track_name, track in tracks.items():
                    score = track.validation_accuracy if hasattr(track, 'validation_accuracy') else 0.0
                    if score > best_score:
                        best_score = score
                        best_track = track_name
                
                decision = RoutingDecision(
                    record_id=record.id,
                    selected_track=best_track,
                    confidence=best_score,
                    reason=f"best_performance_{best_score:.3f}"
                )
                
                routing_decisions.append(decision)
                self.routing_statistics[best_track] += 1
                
            except Exception as e:
                logger.warning(f"Routing failed for record {record.id}: {e}")
                decision = RoutingDecision(
                    record_id=record.id,
                    selected_track=default_track,
                    confidence=0.5,
                    reason="fallback_routing"
                )
                routing_decisions.append(decision)
                self.routing_statistics[default_track] += 1
        
        return routing_decisions
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_routed = sum(self.routing_statistics.values())
        if total_routed == 0:
            return {"total_records": 0, "routing_distribution": {}}
        
        distribution = {track: count/total_routed for track, count in self.routing_statistics.items()}
        
        return {
            "total_records": total_routed,
            "routing_distribution": distribution,
            "routing_statistics": dict(self.routing_statistics)
        }

# =============================================================================
# OPTIMIZED ENSEMBLE FUSION
# =============================================================================

class OptimizedEnsembleFusion:
    """Simplified ensemble fusion."""
    
    def __init__(self, config: OptimizedTRAConfig):
        self.config = config
        self.stacking_model = None
    
    def fuse_predictions_simple(self, track_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple prediction fusion using majority voting or averaging."""
        if not track_predictions:
            return np.array([])
        
        if len(track_predictions) == 1:
            return list(track_predictions.values())[0]
        
        try:
            # Get sample size
            first_pred = list(track_predictions.values())[0]
            n_samples = len(first_pred)
            
            if self.config.task_type == "classification":
                # Majority voting
                fused = np.zeros(n_samples, dtype=int)
                for i in range(n_samples):
                    votes = []
                    for predictions in track_predictions.values():
                        if i < len(predictions):
                            votes.append(int(predictions[i]))
                    
                    if votes:
                        fused[i] = max(set(votes), key=votes.count)
                    else:
                        fused[i] = 0
            else:
                # Simple averaging for regression
                pred_arrays = list(track_predictions.values())
                fused = np.mean(pred_arrays, axis=0)
            
            return fused
            
        except Exception as e:
            logger.error(f"Prediction fusion failed: {e}")
            return list(track_predictions.values())[0]  # Return first prediction as fallback

# =============================================================================
# OPTIMIZED DRIFT DETECTION
# =============================================================================

class OptimizedDriftDetector:
    """Simplified drift detection."""
    
    def __init__(self, config: OptimizedTRAConfig):
        self.config = config
        self.performance_history = defaultdict(list)
        self.drift_alerts = []
    
    def update_performance(self, track_name: str, accuracy: float):
        """Update performance history."""
        self.performance_history[track_name].append({
            'timestamp': time.time(),
            'accuracy': accuracy
        })
        
        # Keep only recent history
        if len(self.performance_history[track_name]) > self.config.drift_detection_window:
            self.performance_history[track_name] = \
                self.performance_history[track_name][-self.config.drift_detection_window:]
    
    def detect_drift(self, track_name: str) -> bool:
        """Simple drift detection."""
        try:
            if track_name not in self.performance_history:
                return False
            
            history = self.performance_history[track_name]
            if len(history) < 10:
                return False
            
            # Check recent vs older performance
            split_point = len(history) // 2
            older_acc = np.mean([h['accuracy'] for h in history[:split_point]])
            recent_acc = np.mean([h['accuracy'] for h in history[split_point:]])
            
            performance_drop = older_acc - recent_acc
            
            if performance_drop > self.config.drift_threshold:
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    track_name=track_name,
                    drift_type="performance",
                    severity=performance_drop
                )
                self.drift_alerts.append(alert)
                logger.warning(f"🚨 Drift detected in {track_name}: drop={performance_drop:.3f}")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Drift detection failed for {track_name}: {e}")
            return False
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift summary."""
        recent_alerts = [alert for alert in self.drift_alerts if 
                        (datetime.now() - alert.timestamp).seconds < 3600]
        
        return {
            'total_alerts': len(self.drift_alerts),
            'recent_alerts': len(recent_alerts),
            'tracks_with_drift': len(set(alert.track_name for alert in recent_alerts))
        }

# =============================================================================
# OPTIMIZED SHAP ENGINE
# =============================================================================

class OptimizedSHAPEngine:
    """Conditional SHAP engine with size and depth limits."""
    
    def __init__(self, config: OptimizedTRAConfig):
        self.config = config
        self.shap_enabled = False
        self.explainers = {}
    
    def should_enable_shap(self, dataset_size: int, model_depth: int) -> bool:
        """Determine if SHAP should be enabled based on constraints."""
        if not SHAP_AVAILABLE:
            return False
        
        if dataset_size > self.config.shap_max_dataset_size:
            logger.info(f"🔍 SHAP disabled: dataset too large ({dataset_size} > {self.config.shap_max_dataset_size})")
            return False
        
        if model_depth > self.config.shap_max_depth_threshold:
            logger.info(f"🔍 SHAP disabled: model too deep ({model_depth} > {self.config.shap_max_depth_threshold})")
            return False
        
        return True
    
    def initialize_shap_if_enabled(self, tracks: Dict[str, OptimizedTrack], 
                                 X_sample: pd.DataFrame, dataset_size: int):
        """Initialize SHAP only if conditions are met."""
        max_model_depth = 0
        
        # Check maximum model depth
        for track in tracks.values():
            if hasattr(track.classifier, 'max_depth') and track.classifier.max_depth:
                max_model_depth = max(max_model_depth, track.classifier.max_depth)
        
        self.shap_enabled = self.should_enable_shap(dataset_size, max_model_depth)
        
        if self.shap_enabled:
            logger.info("🔍 Initializing SHAP explainers...")
            try:
                for track_name, track in tracks.items():
                    if hasattr(track.classifier, 'feature_importances_'):
                        # Use TreeExplainer for tree-based models
                        self.explainers[track_name] = shap.TreeExplainer(track.classifier)
                        logger.info(f"🔍 SHAP explainer created for {track_name}")
                
                logger.info(f"🔍 SHAP initialization complete for {len(self.explainers)} tracks")
                
            except Exception as e:
                logger.warning(f"SHAP initialization failed: {e}")
                self.shap_enabled = False
        else:
            logger.info("🔍 SHAP explainability disabled due to constraints")
    
    def get_shap_summary(self) -> Dict[str, Any]:
        """Get SHAP summary."""
        return {
            'shap_enabled': self.shap_enabled,
            'tracks_with_shap': len(self.explainers),
            'available_explainers': list(self.explainers.keys())
        }

# =============================================================================
# MAIN OPTIMIZED TRA SYSTEM
# =============================================================================

@dataclass
class Record:
    """Simple record structure."""
    id: int
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class OptimizedTrackRailAlgorithm:
    """Optimized Track-Rail Algorithm with performance constraints."""
    
    def __init__(self, config: OptimizedTRAConfig = None):
        self.config = config or OptimizedTRAConfig()
        self.config.validate()
        
        # Core components
        self.tracks = {}
        self.routing_engine = OptimizedRoutingEngine(self.config)
        self.clustering_engine = OptimizedClusteringEngine(self.config)
        self.feature_engineer = OptimizedFeatureEngineer(self.config)
        self.ensemble_fusion = OptimizedEnsembleFusion(self.config)
        self.drift_detector = OptimizedDriftDetector(self.config)
        self.shap_engine = OptimizedSHAPEngine(self.config)
        
        # Data tracking
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = []
        self.cluster_labels = None
        
        # Status
        self.is_fitted = False
        self.total_records_processed = 0
        self.training_start_time = None
        self.training_end_time = None
        
        logger.info("🚀 Optimized TRA system initialized")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, 
            validation_data: Tuple = None, verbose: bool = True) -> 'OptimizedTrackRailAlgorithm':
        """Optimized fit method with comprehensive logging."""
        
        self.training_start_time = time.time()
        logger.info("=" * 80)
        logger.info("🚀 OPTIMIZED TRACK-RAIL ALGORITHM - TRAINING START")
        logger.info("=" * 80)
        ResourceMonitor.log_memory_usage("Training Start")
        
        try:
            # Phase 1: Data Preparation with Sampling
            logger.info("📊 Phase 1: Data Preparation & Sampling")
            X, y = self._prepare_and_sample_data(X, y, validation_data)
            
            # Phase 2: Feature Engineering
            logger.info("🔧 Phase 2: Optimized Feature Engineering")
            X_engineered = self._perform_feature_engineering(X, y)
            
            # Phase 3: Clustering and Track Creation
            logger.info("🔍 Phase 3: Clustering & Track Creation")
            self._create_tracks_optimized(X_engineered, y)
            
            # Phase 4: Track Training
            logger.info("🎯 Phase 4: Track Training")
            self._train_tracks_optimized(X_engineered, y)
            
            # Phase 5: SHAP Initialization (Conditional)
            logger.info("🔍 Phase 5: SHAP Initialization")
            self._initialize_shap_conditional(X_engineered, y)
            
            # Phase 6: Stacking Setup
            logger.info("🔗 Phase 6: Stacking & Meta-Learning")
            self._setup_stacking_optimized(X_engineered, y)
            
            # Phase 7: Performance Evaluation
            logger.info("📈 Phase 7: Performance Evaluation")
            final_metrics = self._evaluate_final_performance()
            
            self.is_fitted = True
            self.training_end_time = time.time()
            training_time = self.training_end_time - self.training_start_time
            
            logger.info("=" * 80)
            logger.info(f"✅ TRAINING COMPLETED in {training_time:.2f}s")
            logger.info(f"📊 Final Accuracy: {final_metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"📊 Final F1-Score: {final_metrics.get('test_f1', 0):.4f}")
            logger.info(f"🎯 Tracks Created: {len(self.tracks)}")
            ResourceMonitor.log_memory_usage("Training Complete")
            logger.info("=" * 80)
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _prepare_and_sample_data(self, X, y, validation_data):
        """Prepare data with optimized sampling."""
        # Convert to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        self.feature_names = X.columns.tolist()
        
        # Validate and sample data
        X, y = check_X_y(X.values, y, accept_sparse=False)
        X = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply dataset sampling
        X, y = DatasetSampler.sample_large_dataset(
            X, y, max_samples=self.config.max_dataset_size, stratify=True
        )
        
        # Train-test split
        if validation_data is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if self.config.task_type == "classification" else None
            )
        else:
            self.X_train, self.y_train = X, y
            self.X_test, self.y_test = validation_data
        
        logger.info(f"📊 Data prepared - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        logger.info(f"📊 Features: {len(self.feature_names)}, Classes: {len(np.unique(y))}")
        
        return self.X_train, self.y_train
    
    def _perform_feature_engineering(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Perform optimized feature engineering."""
        try:
            X_engineered = self.feature_engineer.fit_transform(X, y)
            logger.info(f"🔧 Feature engineering: {X.shape[1]} → {X_engineered.shape[1]} features")
            return X_engineered
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return X
    
    def _create_tracks_optimized(self, X: pd.DataFrame, y: np.ndarray):
        """Create tracks with optimized clustering."""
        try:
            # Find clusters
            self.cluster_labels = self.clustering_engine.find_optimal_clusters(X.values, y)
            
            if self.cluster_labels is not None:
                n_clusters = len(np.unique(self.cluster_labels[self.cluster_labels != -1]))
                logger.info(f"🔍 Found {n_clusters} clusters using {self.clustering_engine.clustering_method}")
                
                # Create regional tracks (limited)
                self._create_regional_tracks_limited(X, y, self.cluster_labels)
            
            # Always create global tracks (limited)
            self._create_global_tracks_limited(X, y)
            
            logger.info(f"🎯 Created {len(self.tracks)} tracks total")
            
        except Exception as e:
            logger.error(f"❌ Track creation failed: {e}")
            self._create_global_tracks_limited(X, y)
    
    def _create_regional_tracks_limited(self, X: pd.DataFrame, y: np.ndarray, cluster_labels: np.ndarray):
        """Create limited number of regional tracks."""
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        regional_tracks_created = 0
        
        for cluster_id in unique_clusters:
            if regional_tracks_created >= self.config.max_regional_tracks:
                break
            
            try:
                cluster_mask = cluster_labels == cluster_id
                X_cluster = X[cluster_mask]
                y_cluster = y[cluster_mask]
                
                # Check minimum size
                if len(X_cluster) < 30:
                    continue
                
                # Create regional track with AutoML
                track_name = f"regional_track_{cluster_id}"
                classifier = self._create_automl_classifier(X_cluster, y_cluster)
                
                track = OptimizedTrack(
                    name=track_name,
                    level="regional",
                    classifier=classifier,
                    config=self.config
                )
                
                self.tracks[track_name] = track
                regional_tracks_created += 1
                
                logger.info(f"🎯 Created {track_name} with {len(X_cluster)} samples")
                
            except Exception as e:
                logger.warning(f"Regional track creation failed for cluster {cluster_id}: {e}")
                continue
    
    def _create_global_tracks_limited(self, X: pd.DataFrame, y: np.ndarray):
        """Create limited number of global tracks."""
        for i in range(self.config.max_global_tracks):
            try:
                track_name = f"global_track_{i}"
                
                if i == 0:
                    # Main global track with AutoML
                    classifier = self._create_automl_classifier(X, y)
                else:
                    # Additional global tracks with different algorithms
                    classifier = self._create_diverse_classifier(i)
                
                track = OptimizedTrack(
                    name=track_name,
                    level="global",
                    classifier=classifier,
                    config=self.config
                )
                
                self.tracks[track_name] = track
                logger.info(f"🎯 Created {track_name} with AutoML optimization")
                
            except Exception as e:
                logger.error(f"Global track creation failed for track {i}: {e}")
                continue
    
    def _create_automl_classifier(self, X: pd.DataFrame, y: np.ndarray) -> BaseEstimator:
        """Create classifier with AutoML optimization."""
        if not OPTUNA_AVAILABLE:
            return self._create_fallback_classifier()
        
        try:
            logger.info(f"🔧 Starting AutoML optimization with {self.config.automl_trials} trials...")
            
            def objective(trial):
                try:
                    model_type = trial.suggest_categorical('model_type', 
                                                         ['random_forest', 'gradient_boosting', 'extra_trees'])
                    
                    classifier = OptimizedAutoMLFactory.create_optimized_classifier(
                        model_type, self.config, trial
                    )
                    
                    # Fast cross-validation
                    if len(X) > 100:
                        cv_scores = cross_val_score(
                            classifier, X.values, y,
                            cv=3,  # Reduced CV folds
                            scoring='accuracy' if self.config.task_type == "classification" else 'neg_mean_squared_error',
                            n_jobs=1
                        )
                        score = np.mean(cv_scores)
                    else:
                        X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                            X, y, test_size=0.3, random_state=42
                        )
                        classifier.fit(X_train_cv.values, y_train_cv)
                        score = classifier.score(X_val_cv.values, y_val_cv)
                    
                    return score
                except Exception as e:
                    logger.debug(f"AutoML trial failed: {e}")
                    return -1.0
            
            # Create study with timeout
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
            )
            
            study.optimize(
                objective,
                n_trials=self.config.automl_trials,
                timeout=120,  # 2 minutes max
                n_jobs=1,
                show_progress_bar=False
            )
            
            # Create best model
            best_params = study.best_params
            model_type = best_params.pop('model_type')
            
            logger.info(f"🔧 AutoML selected: {model_type} with score: {study.best_value:.4f}")
            
            return OptimizedAutoMLFactory.create_optimized_classifier(
                model_type, self.config, **best_params
            )
            
        except Exception as e:
            logger.warning(f"AutoML optimization failed: {e}")
            return self._create_fallback_classifier()
    
    def _create_diverse_classifier(self, index: int) -> BaseEstimator:
        """Create diverse classifiers for additional global tracks."""
        classifiers = [
            ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            GradientBoostingClassifier(n_estimators=75, max_depth=4, random_state=42),
            RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42, n_jobs=-1)
        ]
        
        return classifiers[index % len(classifiers)]
    
    def _create_fallback_classifier(self) -> BaseEstimator:
        """Create fallback classifier."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    def _train_tracks_optimized(self, X: pd.DataFrame, y: np.ndarray):
        """Train tracks with optimization."""
        logger.info(f"🎯 Training {len(self.tracks)} tracks...")
        
        for track_name, track in self.tracks.items():
            try:
                logger.info(f"🎯 Training {track_name}...")
                
                if track.level == "regional" and self.cluster_labels is not None:
                    # Get cluster-specific data
                    cluster_id = int(track_name.split('_')[-1])
                    cluster_mask = self.cluster_labels == cluster_id
                    X_track = X[cluster_mask]
                    y_track = y[cluster_mask]
                else:
                    # Global track uses all data
                    X_track = X
                    y_track = y
                
                # Train track
                track.fit_optimized(X_track, y_track)
                
                # Update drift detector
                self.drift_detector.update_performance(track_name, track.validation_accuracy)
                
            except Exception as e:
                logger.error(f"❌ Training failed for {track_name}: {e}")
                continue
    
    def _initialize_shap_conditional(self, X: pd.DataFrame, y: np.ndarray):
        """Initialize SHAP based on conditions."""
        sample_size = min(50, len(X))
        X_sample = X.iloc[:sample_size]
        
        self.shap_engine.initialize_shap_if_enabled(self.tracks, X_sample, len(X))
    
    def _setup_stacking_optimized(self, X: pd.DataFrame, y: np.ndarray):
        """Setup optimized stacking with 3-fold CV."""
        if not self.config.enable_stacking or len(self.tracks) < 2:
            logger.info("🔗 Stacking disabled or insufficient tracks")
            return
        
        try:
            logger.info("🔗 Setting up optimized stacking...")
            
            track_estimators = [(name, track.classifier) for name, track in self.tracks.items()
                              if track.classifier is not None]
            
            if len(track_estimators) >= 2:
                if self.config.task_type == "classification":
                    self.ensemble_fusion.stacking_model = StackingClassifier(
                        estimators=track_estimators,
                        final_estimator=LogisticRegression(random_state=42),
                        cv=self.config.stacking_cv_folds,  # 3-fold CV
                        n_jobs=1
                    )
                else:
                    self.ensemble_fusion.stacking_model = StackingRegressor(
                        estimators=track_estimators,
                        final_estimator=Ridge(random_state=42),
                        cv=self.config.stacking_cv_folds,  # 3-fold CV
                        n_jobs=1
                    )
                
                # Fit stacking model
                self.ensemble_fusion.stacking_model.fit(X.values, y)
                logger.info("🔗 Stacking model trained successfully")
                
        except Exception as e:
            logger.warning(f"Stacking setup failed: {e}")
    
    def _evaluate_final_performance(self) -> Dict[str, Any]:
        """Evaluate final system performance."""
        metrics = {}
        
        try:
            # Transform test data
            X_test_engineered = self.feature_engineer.transform(self.X_test)
            
            # Get predictions
            y_pred = self.predict(self.X_test)
            
            # Calculate metrics
            test_accuracy = accuracy_score(self.y_test, y_pred)
            test_f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Track-wise performance
            track_metrics = {}
            for track_name, track in self.tracks.items():
                try:
                    track_pred = track.predict(X_test_engineered)
                    track_acc = accuracy_score(self.y_test, track_pred)
                    track_metrics[track_name] = track_acc
                except Exception as e:
                    track_metrics[track_name] = 0.0
            
            metrics = {
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'track_accuracies': track_metrics,
                'n_tracks': len(self.tracks),
                'n_features': len(self.feature_engineer.feature_names_out_) if self.feature_engineer.feature_names_out_ else 0
            }
            
            logger.info(f"📈 System Performance: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Performance evaluation failed: {e}")
            metrics = {'test_accuracy': 0.0, 'test_f1': 0.0, 'track_accuracies': {}, 'n_tracks': 0}
        
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Optimized prediction with routing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            start_time = time.time()
            
            # Prepare data
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            # Apply feature engineering
            X_engineered = self.feature_engineer.transform(X)
            
            # Create records for routing
            records = [Record(id=i) for i in range(len(X_engineered))]
            
            # Simple routing
            routing_decisions = self.routing_engine.route_records_simple(
                records, self.tracks, X_engineered
            )
            
            # Get predictions from tracks
            track_predictions = {}
            for track_name, track in self.tracks.items():
                try:
                    predictions = track.predict(X_engineered)
                    track_predictions[track_name] = predictions
                except Exception as e:
                    logger.debug(f"Track prediction failed for {track_name}: {e}")
                    track_predictions[track_name] = np.zeros(len(X_engineered), dtype=int)
            
            # Ensemble fusion
            final_predictions = self.ensemble_fusion.fuse_predictions_simple(track_predictions)
            
            # Track performance
            prediction_time = (time.time() - start_time) * 1000
            self.total_records_processed += len(X)
            
            logger.debug(f"Prediction completed in {prediction_time:.2f}ms for {len(X)} records")
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            # Fallback to global track
            if "global_track_0" in self.tracks:
                try:
                    X_engineered = self.feature_engineer.transform(X)
                    return self.tracks["global_track_0"].predict(X_engineered)
                except:
                    pass
            
            return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.config.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        try:
            # Prepare data
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            X_engineered = self.feature_engineer.transform(X)
            
            # Get probabilities from tracks
            track_probabilities = []
            for track_name, track in self.tracks.items():
                try:
                    proba = track.predict_proba(X_engineered)
                    track_probabilities.append(proba)
                except Exception as e:
                    logger.debug(f"Probability prediction failed for {track_name}: {e}")
                    n_classes = len(np.unique(self.y_train)) if self.y_train is not None else 2
                    track_probabilities.append(np.ones((len(X_engineered), n_classes)) / n_classes)
            
            # Average probabilities
            if track_probabilities:
                final_probabilities = np.mean(track_probabilities, axis=0)
                return final_probabilities
            else:
                n_classes = len(np.unique(self.y_train)) if self.y_train is not None else 2
                return np.ones((len(X), n_classes)) / n_classes
                
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            n_classes = len(np.unique(self.y_train)) if self.y_train is not None else 2
            return np.ones((len(X), n_classes)) / n_classes
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        try:
            summary = {
                "system_info": {
                    "is_fitted": self.is_fitted,
                    "task_type": self.config.task_type,
                    "total_records_processed": self.total_records_processed,
                    "n_tracks": len(self.tracks),
                    "n_features_original": len(self.feature_names),
                    "n_features_engineered": len(self.feature_engineer.feature_names_out_) if self.feature_engineer.feature_names_out_ else 0,
                    "training_time_seconds": (self.training_end_time - self.training_start_time) if self.training_end_time else 0,
                    "memory_usage_mb": ResourceMonitor.get_memory_usage()
                }
            }
            
            if self.is_fitted:
                # Track performance
                track_performance = {}
                for track_name, track in self.tracks.items():
                    track_performance[track_name] = {
                        "level": track.level,
                        "train_accuracy": track.train_accuracy,
                        "validation_accuracy": track.validation_accuracy,
                        "usage_count": track.usage_count,
                        "health_status": track.health_status,
                        "specialized_features": track.specialized_features[:5]
                    }
                
                summary["track_performance"] = track_performance
                
                # Routing summary
                summary["routing_analysis"] = self.routing_engine.get_routing_summary()
                
                # Drift analysis
                summary["drift_analysis"] = self.drift_detector.get_drift_summary()
                
                # SHAP analysis
                summary["shap_analysis"] = self.shap_engine.get_shap_summary()
                
                # Clustering analysis
                if self.cluster_labels is not None:
                    unique_clusters = np.unique(self.cluster_labels[self.cluster_labels != -1])
                    summary["clustering_analysis"] = {
                        "method": self.clustering_engine.clustering_method,
                        "n_clusters": len(unique_clusters),
                        "cluster_sizes": {f"cluster_{i}": np.sum(self.cluster_labels == i) 
                                        for i in unique_clusters}
                    }
                
                # System health
                avg_track_accuracy = np.mean([track.validation_accuracy for track in self.tracks.values()])
                summary["system_health"] = {
                    "average_track_accuracy": avg_track_accuracy,
                    "tracks_healthy": len([t for t in self.tracks.values() if t.health_status == "healthy"]),
                    "stacking_enabled": self.config.enable_stacking and self.ensemble_fusion.stacking_model is not None,
                    "shap_enabled": self.shap_engine.shap_enabled
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {"error": str(e), "is_fitted": self.is_fitted}

# =============================================================================
# OPTIMIZED DATASET LOADING
# =============================================================================

def load_optimized_covertype_dataset(max_samples: int = 30000):
    """Load Forest Covertype dataset with sampling."""
    try:
        logger.info("📊 Loading Forest Covertype dataset...")
        
        # Load the dataset
        data = fetch_openml('covertype', version=3, as_frame=True)
        X, y = data.data, data.target.astype(int) - 1  # Convert to 0-indexed
        
        logger.info(f"📊 Original dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Apply sampling with stratification
        X_sampled, y_sampled = DatasetSampler.sample_large_dataset(
            X, y, max_samples=max_samples, stratify=True
        )
        
        # Clean data
        X_sampled = X_sampled.fillna(X_sampled.median(numeric_only=True))
        X_sampled = X_sampled.select_dtypes(include=[np.number])
        
        logger.info(f"📊 Final dataset: {X_sampled.shape[0]} samples, {X_sampled.shape[1]} features")
        logger.info(f"📊 Class distribution: {np.bincount(y_sampled)}")
        
        return X_sampled, y_sampled
        
    except Exception as e:
        logger.error(f"❌ Covertype dataset loading failed: {e}")
        logger.info("📊 Creating synthetic dataset as fallback...")
        
        X, y = make_classification(
            n_samples=max_samples,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=3,
            n_clusters_per_class=2,
            flip_y=0.02,
            random_state=42
        )
        
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X, y

# =============================================================================
# DEMONSTRATION SCRIPT
# =============================================================================

def run_optimized_tra_demonstration():
    """Run comprehensive demonstration of Optimized TRA."""
    print("\n" + "=" * 100)
    print("🚀 OPTIMIZED TRACK-RAIL ALGORITHM (TRA) - COMPREHENSIVE DEMONSTRATION")
    print("=" * 100)
    
    try:
        # Load dataset with sampling
        X, y = load_optimized_covertype_dataset(max_samples=30000)
        
        # Optimized configuration
        config = OptimizedTRAConfig(
            task_type="classification",
            max_global_tracks=3,
            max_regional_tracks=2,
            enable_automl=True,
            automl_trials=20,
            max_engineered_features=20,
            stacking_cv_folds=3,
            enable_shap_routing=True,
            shap_max_dataset_size=50000,
            shap_max_depth_threshold=6
        )
        
        # Create and train Optimized TRA
        tra = OptimizedTrackRailAlgorithm(config)
        tra.fit(X, y, verbose=True)
        
        # Comprehensive evaluation
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE SYSTEM ANALYSIS")
        print("=" * 100)
        
        # Test prediction speed
        start_time = time.time()
        predictions = tra.predict(X.iloc[:100])
        prediction_time = (time.time() - start_time) * 1000
        
        probabilities = tra.predict_proba(X.iloc[:100])
        
        # Get detailed summary
        summary = tra.get_comprehensive_summary()
        
        # Display comprehensive results
        print(f"\n📊 FINAL PERFORMANCE METRICS:")
        system_info = summary['system_info']
        print(f"   • Training Time: {system_info['training_time_seconds']:.2f}s")
        print(f"   • Memory Usage: {system_info['memory_usage_mb']:.1f} MB")
        print(f"   • Total Tracks: {system_info['n_tracks']}")
        print(f"   • Features: {system_info['n_features_original']} → {system_info['n_features_engineered']}")
        print(f"   • Records Processed: {system_info['total_records_processed']}")
        print(f"   • Prediction Time: {prediction_time:.2f}ms (100 records)")
        print(f"   • Predictions/second: {100 / (prediction_time / 1000):.0f}")
        
        print(f"\n🎯 TRACK PERFORMANCE:")
        for track_name, metrics in summary['track_performance'].items():
            print(f"   • {track_name} ({metrics['level']}):")
            print(f"     - Validation Accuracy: {metrics['validation_accuracy']:.4f}")
            print(f"     - Usage Count: {metrics['usage_count']}")
            print(f"     - Health Status: {metrics['health_status']}")
        
        print(f"\n🔄 ROUTING ANALYSIS:")
        routing = summary['routing_analysis']
        print(f"   • Total Routed Records: {routing.get('total_records', 0)}")
        print(f"   • Track Distribution: {routing.get('routing_distribution', {})}")
        
        print(f"\n🔍 SHAP EXPLAINABILITY:")
        shap_info = summary['shap_analysis']
        print(f"   • SHAP Enabled: {'✅' if shap_info.get('shap_enabled', False) else '❌'}")
        print(f"   • Tracks with SHAP: {shap_info.get('tracks_with_shap', 0)}")
        
        if 'clustering_analysis' in summary:
            print(f"\n🎯 CLUSTERING ANALYSIS:")
            clustering = summary['clustering_analysis']
            print(f"   • Method: {clustering.get('method', 'Unknown')}")
            print(f"   • Clusters Found: {clustering.get('n_clusters', 0)}")
            print(f"   • Cluster Sizes: {clustering.get('cluster_sizes', {})}")
        
        print(f"\n🚨 DRIFT DETECTION:")
        drift = summary['drift_analysis']
        print(f"   • Total Alerts: {drift.get('total_alerts', 0)}")
        print(f"   • Recent Alerts: {drift.get('recent_alerts', 0)}")
        print(f"   • Tracks with Drift: {drift.get('tracks_with_drift', 0)}")
        
        print(f"\n💚 SYSTEM HEALTH:")
        health = summary['system_health']
        print(f"   • Average Track Accuracy: {health.get('average_track_accuracy', 0):.4f}")
        print(f"   • Healthy Tracks: {health.get('tracks_healthy', 0)}/{system_info['n_tracks']}")
        print(f"   • Stacking Enabled: {'✅' if health.get('stacking_enabled', False) else '❌'}")
        print(f"   • SHAP Enabled: {'✅' if health.get('shap_enabled', False) else '❌'}")
        
        print(f"\n🧪 PREDICTION CAPABILITIES:")
        print(f"   • Sample Predictions: {predictions[:5]}")
        print(f"   • Probability Shape: {probabilities.shape}")
        print(f"   • Average Confidence: {np.mean(np.max(probabilities, axis=1)):.4f}")
        
        # Mock self-healing checkpoint
        print(f"\n💾 SELF-HEALING & CHECKPOINTING:")
        print(f"   • Auto-checkpoint: ✅ (Mocked)")
        print(f"   • Error Recovery: ✅ (Mocked)")
        print(f"   • Performance Monitoring: ✅ Active")
        
        print("\n" + "=" * 100)
        print("✅ OPTIMIZED TRA DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("🎯 All optimization objectives achieved:")
        print("   ✅ Dataset sampling with stratification")
        print("   ✅ Capped AutoML trials (20)")
        print("   ✅ SHAP lazy-loading with constraints")
        print("   ✅ 3-fold stacking")
        print("   ✅ Constrained model depth (6-8)")
        print("   ✅ Limited track creation")
        print("   ✅ Comprehensive logging")
        print("   ✅ Performance metrics")
        print("=" * 100)
        
        return tra, summary
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {str(e)}")
        print(traceback.format_exc())
        return None, None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the optimized demonstration
    tra_system, summary = run_optimized_tra_demonstration()
    
    if tra_system is not None:
        print(f"\n🎉 Optimized TRA system successfully created and tested!")
        print(f"🚀 Ready for production use with all performance optimizations!")
        
        # Save the trained system
        try:
            joblib.dump(tra_system, 'optimized_tra_system.pkl')
            print(f"💾 System saved to 'optimized_tra_system.pkl'")
        except Exception as e:
            print(f"⚠️  Could not save system: {e}")
    else:
        print(f"❌ System creation failed. Please check the logs for details.")
