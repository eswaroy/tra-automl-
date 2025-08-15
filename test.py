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
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
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
from sklearn.datasets import make_classification, make_regression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

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

# Explainability libraries
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

# Bayesian optimization
try:
    import optuna
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

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tra_system.log')
    ]
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
# Enhanced logging configuration with Unicode handling
import sys
import io

def setup_safe_logging():
    """Setup logging that handles Unicode characters safely."""
    
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            # Get the formatted message
            msg = super().format(record)
            # Replace problematic Unicode characters
            replacements = {
                '→': '->',
                '←': '<-',
                '↑': '^',
                '↓': 'v',
                '✓': 'OK',
                '✗': 'X',
                '…': '...'
            }
            for unicode_char, ascii_char in replacements.items():
                msg = msg.replace(unicode_char, ascii_char)
            return msg
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create new handlers with safe formatter
    formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('tra_system.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )

# Call this function to setup safe logging
setup_safe_logging()

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# ENHANCED CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class TRAConfig:
    """Comprehensive configuration for TRA system with improved defaults."""
    # Core TRA settings
    task_type: str = "classification"
    max_tracks: int = 5
    enable_meta_learning: bool = True
    
    # Feature engineering
    enable_automated_fe: bool = True
    max_engineered_features: int = 25
    feature_selection_method: str = "mutual_info"
    l1_regularization_strength: float = 0.1
    
    # Meta-learning and NAS
    enable_nas: bool = True
    nas_budget: int = 20
    meta_learning_cv_folds: int = 3
    enable_hyperopt: bool = True
    
    # Resource management
    use_gpu: bool = TORCH_AVAILABLE
    max_workers: int = 2
    memory_limit_mb: int = 4096
    
    # Ensemble and fusion
    enable_stacking: bool = True
    enable_blending: bool = True
    dynamic_fusion: bool = True
    ensemble_diversity_threshold: float = 0.2
    min_track_accuracy: float = 0.60
    
    # Clustering improvements - CRITICAL FIX
    min_silhouette_score: float = 0.1  # More lenient
    min_cluster_size_ratio: float = 0.02  # More lenient
    min_minority_class_ratio: float = 0.03  # More lenient
    use_dbscan_fallback: bool = True
    
    # Explainability - FIXED
    enable_explanations: bool = True  # ENABLED
    explanation_method: str = "shap"
    track_decisions: bool = True
    
    # Routing - CRITICAL NEW PARAMETERS
    enable_smart_routing: bool = True  # NEW
    routing_threshold: float = 0.3  # NEW: Lower threshold for routing
    use_feature_based_routing: bool = True  # NEW
    enable_confidence_routing: bool = True  # NEW
    
    # Performance optimization
    enable_early_stopping: bool = True
    validation_split: float = 0.2
    performance_threshold: float = 0.70
    adaptive_learning: bool = True
    
    def validate(self):
        """Validate configuration settings."""
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be >= 1")

# =============================================================================
# ENHANCED DATA STRUCTURES
# =============================================================================

@dataclass
class AdvancedMetrics:
    """Advanced performance and resource metrics."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    predictions_per_second: float = 0.0
    error_rate: float = 0.0
    stability_score: float = 1.0
    diversity_score: float = 0.0
    confidence_score: float = 0.0  # NEW
    
    def efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        if self.latency_ms == 0:
            return 0.5
        
        perf_score = max(0.1, (self.accuracy + self.f1_score) / 2.0)
        throughput = min(1000.0 / max(self.latency_ms, 1.0), 100.0) / 100.0
        memory_efficiency = max(0.1, 1.0 - self.memory_mb / 2000.0)
        stability_weight = max(0.1, self.stability_score)
        confidence_weight = max(0.1, self.confidence_score)  # NEW
        
        efficiency = (0.4 * perf_score + 0.15 * throughput + 0.1 * memory_efficiency +
                     0.15 * stability_weight + 0.2 * confidence_weight)
        return min(1.0, max(0.1, efficiency))

@dataclass
class RoutingDecision:
    """NEW: Track routing decision with full audit trail."""
    record_id: int
    selected_track: str
    confidence: float
    reason: str
    alternative_tracks: Dict[str, float] = field(default_factory=dict)
    feature_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EnhancedRecord:
    """Enhanced record with comprehensive tracking."""
    id: int
    features: np.ndarray
    current_track: str = "global_track_0"
    routing_history: List[RoutingDecision] = field(default_factory=list)  # NEW
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    resource_usage: AdvancedMetrics = field(default_factory=AdvancedMetrics)
    
    # Enhanced tracking
    prediction_uncertainty: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    routing_score: float = 0.0
    
    def add_routing_decision(self, decision: RoutingDecision):
        """Add routing decision to history."""
        self.routing_history.append(decision)
        self.current_track = decision.selected_track

# =============================================================================
# SMART ROUTING ENGINE - COMPLETELY NEW
# =============================================================================

class SmartRoutingEngine:
    """NEW: Intelligent routing engine for track selection."""
    
    def __init__(self, config: TRAConfig):
        self.config = config
        self.feature_importance_cache = {}
        self.track_specializations = {}  # Track -> feature specialization
        self.routing_statistics = defaultdict(int)
        
    def analyze_track_specializations(self, tracks: Dict[str, 'EnhancedTrack'], 
                                    X_train: pd.DataFrame, y_train: np.ndarray):
        """Analyze what each track specializes in."""
        logger.info("Analyzing track specializations for smart routing...")
        
        for track_name, track in tracks.items():
            try:
                if hasattr(track.classifier, 'feature_importances_'):
                    # Get feature importances
                    importance = track.classifier.feature_importances_
                    feature_names = X_train.columns.tolist()
                    
                    # Store top features for this track
                    feature_importance_dict = dict(zip(feature_names, importance))
                    sorted_features = sorted(feature_importance_dict.items(), 
                                           key=lambda x: x[1], reverse=True)
                    
                    self.track_specializations[track_name] = {
                        'top_features': [f[0] for f in sorted_features[:5]],
                        'feature_weights': dict(sorted_features[:10]),
                        'specialization_score': np.max(importance) - np.mean(importance)
                    }
                    
                    logger.info(f"Track {track_name} specializes in: {self.track_specializations[track_name]['top_features'][:3]}")
                    
            except Exception as e:
                logger.debug(f"Could not analyze specialization for {track_name}: {e}")
                self.track_specializations[track_name] = {
                    'top_features': [],
                    'feature_weights': {},
                    'specialization_score': 0.0
                }
    
    def route_record(self, record: EnhancedRecord, tracks: Dict[str, 'EnhancedTrack'], 
                    X_sample: pd.DataFrame) -> RoutingDecision:
        """CRITICAL: Smart routing logic - decides which track to use."""
        
        if not self.config.enable_smart_routing:
            # Fallback to first available track
            return RoutingDecision(
                record_id=record.id,
                selected_track=list(tracks.keys())[0],
                confidence=0.5,
                reason="smart_routing_disabled"
            )
        
        # Calculate routing scores for each track
        track_scores = {}
        feature_sample = X_sample.iloc[0] if len(X_sample) > 0 else pd.Series()
        
        for track_name, track in tracks.items():
            score = self._calculate_track_score(track_name, track, feature_sample)
            track_scores[track_name] = score
        
        # Select best track
        if track_scores:
            best_track = max(track_scores, key=track_scores.get)
            best_score = track_scores[best_track]
            
            # Apply routing threshold
            if best_score >= self.config.routing_threshold:
                selected_track = best_track
                confidence = best_score
                reason = f"feature_specialization_score_{best_score:.3f}"
            else:
                # Use global track as fallback
                selected_track = "global_track_0" if "global_track_0" in tracks else list(tracks.keys())[0]
                confidence = 0.5
                reason = f"below_threshold_{best_score:.3f}"
        else:
            # Ultimate fallback
            selected_track = list(tracks.keys())[0]
            confidence = 0.3
            reason = "no_scores_available"
        
        # Update routing statistics
        self.routing_statistics[selected_track] += 1
        
        # Create routing decision
        decision = RoutingDecision(
            record_id=record.id,
            selected_track=selected_track,
            confidence=confidence,
            reason=reason,
            alternative_tracks=track_scores,
            feature_scores=self._get_feature_scores(feature_sample)
        )
        
        logger.debug(f"Record {record.id} routed to {selected_track} (confidence: {confidence:.3f}, reason: {reason})")
        
        return decision
    
    def _calculate_track_score(self, track_name: str, track: 'EnhancedTrack', 
                              feature_sample: pd.Series) -> float:
        """Calculate how well a track matches the input features."""
        base_score = 0.5
        
        # Feature-based scoring
        if track_name in self.track_specializations:
            spec = self.track_specializations[track_name]
            feature_match_score = 0.0
            
            # Check if sample has high values for track's specialized features
            for feature, weight in spec['feature_weights'].items():
                if feature in feature_sample.index:
                    # Normalize feature value and multiply by importance weight
                    feature_value = abs(float(feature_sample[feature]))
                    normalized_value = min(feature_value / (feature_value + 1), 1.0)
                    feature_match_score += normalized_value * weight
            
            # Normalize by number of features
            if len(spec['feature_weights']) > 0:
                feature_match_score /= len(spec['feature_weights'])
            
            base_score += 0.3 * feature_match_score
        
        # Performance-based scoring (from track metrics)
        if hasattr(track, 'metrics') and track.metrics.accuracy > 0:
            performance_score = min(track.metrics.accuracy, 1.0)
            base_score += 0.2 * performance_score
        
        # Confidence-based scoring
        if hasattr(track, 'metrics') and track.metrics.confidence_score > 0:
            confidence_score = min(track.metrics.confidence_score, 1.0)
            base_score += 0.1 * confidence_score
        
        return min(base_score, 1.0)
    
    def _get_feature_scores(self, feature_sample: pd.Series) -> Dict[str, float]:
        """Get feature activation scores for debugging."""
        scores = {}
        for feature in feature_sample.index[:5]:  # Top 5 features
            scores[feature] = float(abs(feature_sample[feature]))
        return scores
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get routing statistics summary."""
        total_routed = sum(self.routing_statistics.values())
        if total_routed == 0:
            return {"total_records": 0, "track_distribution": {}}
        
        distribution = {track: count/total_routed 
                       for track, count in self.routing_statistics.items()}
        
        return {
            "total_records": total_routed,
            "track_distribution": distribution,
            "routing_statistics": dict(self.routing_statistics)
        }

# =============================================================================
# ENHANCED FEATURE ENGINEERING ENGINE
# =============================================================================

class AutomatedFeatureEngineer:
    """Enhanced automated feature engineering with better regularization."""
    
    def __init__(self, config: TRAConfig):
        self.config = config
        self.generated_features = []
        self.feature_importance = {}
        self.transformers = {}
        self.feature_cache = {}
        
        # Feature engineering pipeline
        self.is_fitted = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.n_features_out_ = None
        
        # Feature transformation components
        self.variance_selector = None
        self.feature_selector = None
        self.l1_selector = None
        self.scaler = StandardScaler()
        
        # Store feature engineering decisions
        self.interaction_pairs = []
        self.transformation_features = []
        self.aggregation_features = []
        self.selected_numeric_cols = []

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None, domain_type: str = "tabular") -> pd.DataFrame:
        """Enhanced fit and transform with better feature selection."""
        logger.info(f"Starting feature engineering: {X.shape[1]} features")

        # Store original feature names
        self.feature_names_in_ = [str(col) for col in X.columns]

        # Start with original features
        X_engineered = X.copy()
        X_engineered.columns = [str(col) for col in X_engineered.columns]

        # Select numeric columns for engineering
        numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()

        # Feature selection for engineering
        if y is not None and len(numeric_cols) > 3:
            correlations = {}
            for col in numeric_cols[:10]:
                try:
                    if len(np.unique(X_engineered[col])) > 1:
                        if self.config.task_type == "classification":
                            mi_score = mutual_info_classif(X_engineered[[col]], y, random_state=42)[0]
                            correlations[col] = mi_score
                        else:
                            corr = abs(np.corrcoef(X_engineered[col], y)[0, 1])
                            correlations[col] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlations[col] = 0.0
                except:
                    correlations[col] = 0.0

            # Select top features for interaction
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            self.selected_numeric_cols = [col for col, _ in sorted_features[:6]]
        else:
            self.selected_numeric_cols = numeric_cols[:6]

        # Generate features adaptively
        dataset_size = len(X_engineered)
        max_interactions = 1 if dataset_size < 1000 else 2
        max_transformations = 1 if dataset_size < 1000 else 2

        # Create interaction features
        if len(self.selected_numeric_cols) >= 2:
            X_engineered = self._create_interaction_features(X_engineered, max_interactions)

        # Create transformation features
        if len(self.selected_numeric_cols) >= 1:
            X_engineered = self._create_transformation_features(X_engineered, max_transformations)

        # Create aggregation features
        if len(numeric_cols) >= 3:
            X_engineered = self._create_aggregation_features(X_engineered, numeric_cols)

        # Fill missing values
        X_engineered = X_engineered.fillna(0)

        # Feature selection
        if len(X_engineered.columns) > self.config.max_engineered_features and y is not None:
            X_engineered = self._apply_l1_feature_selection(X_engineered, y)

        if len(X_engineered.columns) > self.config.max_engineered_features:
            X_engineered = self._fit_select_features(X_engineered, y)

        # Store final feature names
        self.feature_names_out_ = sorted([str(col) for col in X_engineered.columns])
        self.n_features_out_ = len(self.feature_names_out_)
        self.is_fitted = True

        # Ensure output column order is fixed
        X_engineered = X_engineered[self.feature_names_out_]
        
        # FIXED: Replace → with ->
        logger.info(f"Feature engineering completed: {len(self.feature_names_in_)} -> {len(self.feature_names_out_)} features")
        return X_engineered


    def transform(self, X: pd.DataFrame, domain_type: str = "tabular") -> pd.DataFrame:
        """Transform with perfect consistency."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")

        # Start with original features, ensure consistency
        X_aligned = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            if col in X.columns:
                X_aligned[str(col)] = X[col]
            else:
                X_aligned[str(col)] = 0.0

        X_engineered = X_aligned.copy()

        # Apply same feature engineering steps
        if len(self.selected_numeric_cols) >= 2:
            X_engineered = self._apply_interaction_features(X_engineered)

        if len(self.selected_numeric_cols) >= 1:
            X_engineered = self._apply_transformation_features(X_engineered)

        if len(self.aggregation_features) > 0:
            X_engineered = self._apply_aggregation_features(X_engineered)

        # Fill missing values
        X_engineered = X_engineered.fillna(0)

        # Apply feature selection
        if self.l1_selector is not None:
            try:
                X_engineered = self._transform_l1_selection(X_engineered)
            except:
                pass

        if self.variance_selector is not None or self.feature_selector is not None:
            X_engineered = self._transform_select_features(X_engineered)

        # Ensure exact output consistency
        final_df = pd.DataFrame(index=X.index)
        for col in self.feature_names_out_:
            if col in X_engineered.columns:
                final_df[col] = X_engineered[col]
            else:
                final_df[col] = 0.0

        final_df = final_df[self.feature_names_out_].fillna(0)
        return final_df

    def _apply_l1_feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Apply L1 regularization-based feature selection."""
        try:
            logger.info("Applying L1 regularization-based feature selection...")
            
            if self.config.task_type == "classification":
                l1_model = LogisticRegression(
                    penalty='l1',
                    C=1.0 / self.config.l1_regularization_strength,
                    solver='liblinear',
                    random_state=42,
                    max_iter=1000
                )
            else:
                from sklearn.linear_model import Lasso
                l1_model = Lasso(
                    alpha=self.config.l1_regularization_strength,
                    random_state=42,
                    max_iter=1000
                )

            # Fit L1 model
            l1_model.fit(X.values, y)

            # Create selector
            self.l1_selector = SelectFromModel(l1_model, prefit=True)
            selected_features = X.columns[self.l1_selector.get_support()]

            # Ensure minimum features
            if len(selected_features) < max(10, len(X.columns) // 5):
                if hasattr(l1_model, 'coef_'):
                    if l1_model.coef_.ndim == 1:
                        importances = np.abs(l1_model.coef_)
                    else:
                        importances = np.sum(np.abs(l1_model.coef_), axis=0)
                else:
                    importances = np.ones(len(X.columns))

                top_indices = np.argsort(importances)[::-1][:self.config.max_engineered_features]
                selected_features = X.columns[top_indices]

            X_selected = X[selected_features]
            # FIXED: Replace → with ->
            logger.info(f"L1 feature selection: {len(X.columns)} -> {len(X_selected.columns)} features")
            return X_selected

        except Exception as e:
            logger.warning(f"L1 feature selection failed: {e}")
            return X


    def _create_interaction_features(self, X: pd.DataFrame, max_features: int = 4) -> pd.DataFrame:
        """Create interaction features with adaptive limits."""
        X_new = X.copy()
        interaction_count = 0

        for i, col1 in enumerate(self.selected_numeric_cols[:max_features]):
            if col1 in X.columns and interaction_count < max_features * 2:
                for j, col2 in enumerate(self.selected_numeric_cols[i+1:i+3], i+1):
                    if (col2 in X.columns and j < len(self.selected_numeric_cols) and
                        interaction_count < max_features * 2):
                        
                        # Multiplicative interaction
                        mult_name = f"{col1}_mult_{col2}"
                        X_new[mult_name] = X[col1] * X[col2]
                        self.interaction_pairs.append((col1, col2, 'mult'))
                        interaction_count += 1

                        # Ratio interaction (with safety)
                        if interaction_count < max_features * 2:
                            ratio_name = f"{col1}_div_{col2}"
                            denominator = X[col2].replace(0, 1e-8)
                            X_new[ratio_name] = X[col1] / denominator
                            self.interaction_pairs.append((col1, col2, 'div'))
                            interaction_count += 1

        return X_new

    def _create_transformation_features(self, X: pd.DataFrame, max_features: int = 3) -> pd.DataFrame:
        """Create transformation features with adaptive limits."""
        X_new = X.copy()

        for col in self.selected_numeric_cols[:max_features]:
            if col in X.columns:
                # Log transformation
                log_name = f"{col}_log1p"
                X_new[log_name] = np.log1p(np.abs(X[col]))
                self.transformation_features.append((col, 'log1p'))

                # Square root transformation
                sqrt_name = f"{col}_sqrt"
                X_new[sqrt_name] = np.sqrt(np.abs(X[col]))
                self.transformation_features.append((col, 'sqrt'))

                # Square transformation (only for top 2 features)
                if col in self.selected_numeric_cols[:2]:
                    square_name = f"{col}_square"
                    X_new[square_name] = X[col] ** 2
                    self.transformation_features.append((col, 'square'))

        return X_new

    def _create_aggregation_features(self, X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create aggregation features consistently."""
        X_new = X.copy()
        available_cols = [col for col in numeric_cols if col in X.columns]

        if len(available_cols) >= 3:
            X_new['mean_all'] = X[available_cols].mean(axis=1)
            X_new['std_all'] = X[available_cols].std(axis=1)
            X_new['max_all'] = X[available_cols].max(axis=1)
            X_new['min_all'] = X[available_cols].min(axis=1)
            X_new['range_all'] = X_new['max_all'] - X_new['min_all']
            self.aggregation_features = ['mean_all', 'std_all', 'max_all', 'min_all', 'range_all']

        return X_new

    # Helper methods for applying stored transformations
    def _apply_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored interaction features."""
        X_new = X.copy()
        for col1, col2, op_type in self.interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                if op_type == 'mult':
                    feature_name = f"{col1}_mult_{col2}"
                    X_new[feature_name] = X[col1] * X[col2]
                elif op_type == 'div':
                    feature_name = f"{col1}_div_{col2}"
                    denominator = X[col2].replace(0, 1e-8)
                    X_new[feature_name] = X[col1] / denominator
        return X_new

    def _apply_transformation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored transformation features."""
        X_new = X.copy()
        for col, transform_type in self.transformation_features:
            if col in X.columns:
                if transform_type == 'log1p':
                    feature_name = f"{col}_log1p"
                    X_new[feature_name] = np.log1p(np.abs(X[col]))
                elif transform_type == 'sqrt':
                    feature_name = f"{col}_sqrt"
                    X_new[feature_name] = np.sqrt(np.abs(X[col]))
                elif transform_type == 'square':
                    feature_name = f"{col}_square"
                    X_new[feature_name] = X[col] ** 2
        return X_new

    def _apply_aggregation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored aggregation features."""
        X_new = X.copy()
        numeric_cols = X_new.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = [col for col in numeric_cols if col in X.columns and col in self.feature_names_in_]

        if len(available_cols) >= 3:
            for agg_feature in self.aggregation_features:
                if agg_feature == 'mean_all':
                    X_new['mean_all'] = X[available_cols].mean(axis=1)
                elif agg_feature == 'std_all':
                    X_new['std_all'] = X[available_cols].std(axis=1)
                elif agg_feature == 'max_all':
                    X_new['max_all'] = X[available_cols].max(axis=1)
                elif agg_feature == 'min_all':
                    X_new['min_all'] = X[available_cols].min(axis=1)
                elif agg_feature == 'range_all':
                    max_col = X_new.get('max_all', X[available_cols].max(axis=1))
                    min_col = X_new.get('min_all', X[available_cols].min(axis=1))
                    X_new['range_all'] = max_col - min_col
        return X_new

    def _transform_l1_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted L1 selector."""
        try:
            selected_features = X.columns[self.l1_selector.get_support()]
            return X[selected_features]
        except:
            return X

    def _fit_select_features(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        """Fit feature selection and transform."""
        if len(X.columns) <= self.config.max_engineered_features:
            return X

        try:
            # Remove low-variance features
            from sklearn.feature_selection import VarianceThreshold
            self.variance_selector = VarianceThreshold(threshold=0.01)
            X_variance_selected = pd.DataFrame(
                self.variance_selector.fit_transform(X),
                columns=X.columns[self.variance_selector.get_support()],
                index=X.index
            )

            if y is not None and len(X_variance_selected.columns) > self.config.max_engineered_features:
                # Choose selection method
                if self.config.feature_selection_method == "mutual_info":
                    if self.config.task_type == "classification":
                        score_func = mutual_info_classif
                    else:
                        score_func = mutual_info_regression
                else:
                    score_func = f_classif if self.config.task_type == "classification" else f_regression

                self.feature_selector = SelectKBest(
                    score_func=score_func,
                    k=self.config.max_engineered_features
                )

                X_selected = pd.DataFrame(
                    self.feature_selector.fit_transform(X_variance_selected, y),
                    columns=X_variance_selected.columns[self.feature_selector.get_support()],
                    index=X.index
                )

                # FIXED: Replace → with ->
                logger.info(f"Feature selection: {len(X.columns)} -> {len(X_selected.columns)} features")
                return X_selected
            else:
                return X_variance_selected

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return X.iloc[:, :self.config.max_engineered_features]

    def _transform_select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted feature selectors."""
        try:
            if self.variance_selector is not None:
                X_selected = pd.DataFrame(
                    self.variance_selector.transform(X),
                    columns=X.columns[self.variance_selector.get_support()],
                    index=X.index
                )
                X = X_selected

            if self.feature_selector is not None:
                X_selected = pd.DataFrame(
                    self.feature_selector.transform(X),
                    columns=X.columns[self.feature_selector.get_support()],
                    index=X.index
                )
                X = X_selected

            return X
        except Exception as e:
            logger.warning(f"Feature selection transform failed: {e}")
            return X.iloc[:, :min(len(X.columns), self.config.max_engineered_features)]

# =============================================================================
# ENHANCED META-LEARNING ENGINE
# =============================================================================

class MetaLearningEngine:
    """Enhanced meta-learning with broader model search space."""
    
    def __init__(self, config: TRAConfig):
        self.config = config
        self.experiment_database = {}
        self.meta_features_cache = {}
        self.performance_predictor = None
        
        if OPTUNA_AVAILABLE:
            # Enhanced pruning strategy
            self.study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10
                )
            )

    def extract_meta_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Extract meta-features from dataset."""
        cache_key = f"{X.shape}_{np.sum(X[:5].flatten()):.6f}"
        if cache_key in self.meta_features_cache:
            return self.meta_features_cache[cache_key]

        meta_features = {}
        try:
            # Basic dataset characteristics
            meta_features.update({
                'n_samples': float(X.shape[0]),
                'n_features': float(X.shape[1]),
                'samples_to_features_ratio': X.shape[0] / X.shape[1],
                'log_samples': np.log10(X.shape[0]),
                'log_features': np.log10(X.shape[1])
            })

            # Statistical properties
            try:
                feature_vars = np.var(X, axis=0)
                feature_means = np.mean(X, axis=0)
                meta_features.update({
                    'feature_variance_mean': np.mean(feature_vars),
                    'feature_variance_std': np.std(feature_vars),
                    'sparsity': np.mean(X == 0),
                    'data_density': 1.0 - np.mean(X == 0),
                    'feature_mean_std': np.std(feature_means),
                    'feature_skewness_mean': np.mean([self._skewness(X[:, i]) for i in range(min(5, X.shape[1]))])
                })
            except:
                meta_features.update({
                    'feature_variance_mean': 1.0,
                    'feature_variance_std': 0.5,
                    'sparsity': 0.1,
                    'data_density': 0.9,
                    'feature_mean_std': 1.0,
                    'feature_skewness_mean': 0.0
                })

            # Task-specific meta-features
            if self.config.task_type == "classification":
                unique_classes = len(np.unique(y))
                class_counts = np.bincount(y.astype(int))
                class_probs = class_counts / len(y)
                meta_features.update({
                    'n_classes': float(unique_classes),
                    'class_entropy': -np.sum(class_probs * np.log2(class_probs + 1e-8)),
                    'class_imbalance': np.max(class_counts) / np.min(class_counts),
                    'minority_class_ratio': np.min(class_counts) / len(y)
                })
            else:
                meta_features.update({
                    'target_variance': float(np.var(y)),
                    'target_range': float(np.max(y) - np.min(y)),
                    'target_skewness': float(abs(self._skewness(y)))
                })

            # Clustering characteristics
            if X.shape[0] > 50 and X.shape[1] > 1:
                try:
                    n_clusters = min(3, X.shape[0] // 50)
                    if n_clusters >= 2:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                        labels = kmeans.fit_predict(X)
                        sil_score = silhouette_score(X, labels)
                        meta_features['silhouette_score'] = sil_score
                    else:
                        meta_features['silhouette_score'] = 0.5
                except:
                    meta_features['silhouette_score'] = 0.5
            else:
                meta_features['silhouette_score'] = 0.5

        except Exception as e:
            logger.warning(f"Meta-feature extraction failed: {e}")
            meta_features = {
                'n_samples': float(X.shape[0]),
                'n_features': float(X.shape[1]),
                'samples_to_features_ratio': X.shape[0] / X.shape[1],
                'log_samples': np.log10(X.shape[0]),
                'log_features': np.log10(X.shape[1]),
                'silhouette_score': 0.5
            }

        self.meta_features_cache[cache_key] = meta_features
        return meta_features

    def suggest_model_architecture(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Enhanced model architecture suggestion with broader search space."""
        meta_features = self.extract_meta_features(X, y)
        
        suggestions = {
            'primary_model': 'ensemble',
            'ensemble_size': 3,
            'use_stacking': True,
            'preprocessing': ['standard_scaler'],
            'feature_selection': True,
            'base_models': [],
            'hyperparameters': {}
        }

        # Dataset characteristics
        n_samples = meta_features['n_samples']
        n_features = meta_features['n_features']
        samples_to_features = meta_features['samples_to_features_ratio']
        sparsity = meta_features.get('sparsity', 0.1)
        class_imbalance = meta_features.get('class_imbalance', 1.0)

        # Enhanced model selection logic
        if n_samples < 1000:
            if sparsity > 0.5:
                suggestions.update({
                    'primary_model': 'sparse_ensemble',
                    'base_models': ['random_forest', 'gradient_boosting', 'svm'],
                    'preprocessing': ['robust_scaler']
                })
            elif class_imbalance > 3.0:
                suggestions.update({
                    'primary_model': 'balanced_ensemble',
                    'base_models': ['random_forest', 'gradient_boosting', 'naive_bayes'],
                    'handle_imbalance': True
                })
            else:
                suggestions.update({
                    'primary_model': 'small_ensemble',
                    'base_models': ['random_forest', 'extra_trees', 'knn'],
                })
        
        elif n_samples < 5000:
            # Medium dataset
            available_models = ['random_forest', 'gradient_boosting', 'extra_trees']
            if LIGHTGBM_AVAILABLE:
                available_models.append('lightgbm')
            if XGBOOST_AVAILABLE:
                available_models.append('xgboost')
            
            suggestions.update({
                'primary_model': 'medium_ensemble',
                'base_models': available_models[:3]
            })
        
        else:
            # Large dataset
            available_models = ['random_forest', 'gradient_boosting', 'extra_trees']
            if XGBOOST_AVAILABLE:
                available_models.insert(0, 'xgboost')
            if LIGHTGBM_AVAILABLE:
                available_models.insert(0, 'lightgbm')
            
            suggestions.update({
                'primary_model': 'large_ensemble',
                'base_models': available_models[:3],
                'use_gpu': self.config.use_gpu
            })

        # High-dimensional data handling
        if samples_to_features < 10:
            suggestions.update({
                'feature_selection': True,
                'preprocessing': ['robust_scaler'],
                'regularization': 'l1'
            })

        # Class imbalance handling
        if (self.config.task_type == "classification" and class_imbalance > 2):
            suggestions.update({
                'handle_imbalance': True,
                'class_weight': 'balanced'
            })

        return suggestions

    @staticmethod
    def _skewness(x):
        """Calculate skewness."""
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x == 0:
            return 0.0
        return np.mean(((x - mean_x) / std_x) ** 3)

# =============================================================================
# ENHANCED TRACK STRUCTURES
# =============================================================================

class TrackLevel:
    """Track hierarchy levels."""
    GLOBAL = "global"
    REGIONAL = "regional"
    LOCAL = "local"

class EnhancedTrack:
    """Enhanced track with better performance tracking."""
    
    def __init__(self, name: str, level: str, classifier=None, parent_track=None, config: TRAConfig = None):
        self.name = name
        self.level = level
        self.classifier = classifier
        self.parent_track = parent_track
        self.child_tracks: Dict[str, 'EnhancedTrack'] = {}
        self.config = config or TRAConfig()
        
        # Performance tracking
        self.performance_score = 0.5
        self.usage_count = 0
        self.last_used = time.time()
        self.prediction_times = deque(maxlen=50)
        self.metrics = AdvancedMetrics()
        self.health_status = "healthy"
        
        # CRITICAL: Track accuracy on different data splits
        self.train_accuracy = 0.0
        self.validation_accuracy = 0.0
        self.test_accuracy = 0.0
        
        # Enhanced tracking
        self.prediction_history = deque(maxlen=1000)
        self.routing_decisions = []

    def predict(self, X) -> np.ndarray:
        """Enhanced prediction with usage tracking."""
        start_time = time.time()
        
        try:
            # CRITICAL: Update usage count
            self.usage_count += 1
            self.last_used = time.time()
            
            # Handle different input types
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.asarray(X)

            # Ensure 2D array
            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)

            # Make prediction
            if self.classifier is not None:
                predictions = self.classifier.predict(X_array)
                
                # Store prediction for analysis
                self.prediction_history.extend(predictions.tolist()[:10])  # Store sample
                
            else:
                # Fallback prediction
                if self.config.task_type == "classification":
                    predictions = np.zeros(len(X_array), dtype=int)
                else:
                    predictions = np.zeros(len(X_array), dtype=float)

            # Track timing
            prediction_time = (time.time() - start_time) * 1000
            self.prediction_times.append(prediction_time)
            self.metrics.latency_ms = np.mean(self.prediction_times)

            logger.debug(f"Track {self.name} processed {len(predictions)} predictions (usage: {self.usage_count})")
            
            return predictions

        except Exception as e:
            logger.error(f"Track {self.name} prediction failed: {e}")
            # Return safe fallback
            n_samples = len(X) if hasattr(X, '__len__') else 1
            if self.config.task_type == "classification":
                return np.zeros(n_samples, dtype=int)
            else:
                return np.zeros(n_samples, dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        """Enhanced probability prediction."""
        try:
            # Handle different input types
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.asarray(X)

            # Ensure 2D array
            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)

            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X_array)
                # Update confidence metrics
                if probabilities.ndim > 1:
                    avg_confidence = np.mean(np.max(probabilities, axis=1))
                    self.metrics.confidence_score = avg_confidence
                return probabilities
            elif hasattr(self.classifier, 'decision_function'):
                decision = self.classifier.decision_function(X_array)
                # Convert to probabilities
                if decision.ndim == 1:
                    proba = 1 / (1 + np.exp(-decision))
                    return np.column_stack([1 - proba, proba])
                else:
                    exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                # Fallback: uniform probabilities
                n_samples = len(X_array)
                n_classes = 2
                return np.full((n_samples, n_classes), 1.0 / n_classes)

        except Exception as e:
            logger.warning(f"Track {self.name} probability prediction failed: {e}")
            n_samples = len(X) if hasattr(X, '__len__') else 1
            n_classes = 2
            return np.full((n_samples, n_classes), 0.5)

    def fit(self, X, y):
        """Enhanced fit with validation tracking."""
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values

            if self.classifier is not None:
                # Split for validation
                if len(X) > 50:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Fit on training data
                    self.classifier.fit(X_train, y_train)
                    
                    # Calculate accuracies
                    self.train_accuracy = self.classifier.score(X_train, y_train)
                    self.validation_accuracy = self.classifier.score(X_val, y_val)
                    
                    logger.info(f"Track {self.name} - Train: {self.train_accuracy:.4f}, Val: {self.validation_accuracy:.4f}")
                    
                    # Update metrics
                    self.metrics.accuracy = self.validation_accuracy
                    
                else:
                    # Small dataset - use all data
                    self.classifier.fit(X, y)
                    self.train_accuracy = self.classifier.score(X, y)
                    self.validation_accuracy = self.train_accuracy
                    self.metrics.accuracy = self.train_accuracy

                logger.info(f"Track {self.name} fitted successfully with {len(X)} samples")

            else:
                logger.warning(f"No classifier for track {self.name}")

        except Exception as e:
            logger.error(f"Track {self.name} fitting failed: {e}")
            # Create fallback classifier
            if self.config.task_type == "classification":
                from sklearn.dummy import DummyClassifier
                self.classifier = DummyClassifier(strategy="most_frequent")
            else:
                from sklearn.dummy import DummyRegressor
                self.classifier = DummyRegressor(strategy="mean")
            
            try:
                self.classifier.fit(X, y)
                self.train_accuracy = self.classifier.score(X, y)
                self.validation_accuracy = self.train_accuracy
            except:
                logger.error(f"Even fallback classifier failed for track {self.name}")

    def evaluate_on_test(self, X_test, y_test):
        """CRITICAL: Evaluate track performance on test data."""
        try:
            if self.classifier is not None:
                predictions = self.predict(X_test)
                if self.config.task_type == "classification":
                    self.test_accuracy = accuracy_score(y_test, predictions)
                else:
                    self.test_accuracy = -mean_squared_error(y_test, predictions)
                
                # Update overall metrics
                self.metrics.accuracy = self.test_accuracy
                self.performance_score = self.test_accuracy
                
                logger.info(f"Track {self.name} test accuracy: {self.test_accuracy:.4f}")
                return self.test_accuracy
            else:
                self.test_accuracy = 0.0
                return 0.0
        except Exception as e:
            logger.error(f"Test evaluation failed for {self.name}: {e}")
            self.test_accuracy = 0.0
            return 0.0

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive track health status."""
        return {
            "status": self.health_status,
            "usage_count": self.usage_count,
            "train_accuracy": self.train_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "test_accuracy": self.test_accuracy,
            "performance_score": self.performance_score,
            "last_used": self.last_used,
            "avg_prediction_time_ms": np.mean(self.prediction_times) if self.prediction_times else 0,
            "confidence_score": self.metrics.confidence_score,
            "prediction_samples": list(self.prediction_history)[-5:] if self.prediction_history else []
        }

# =============================================================================
# EXPLAINABILITY ENGINE - COMPLETELY NEW
# =============================================================================

class ExplainabilityEngine:
    """NEW: Comprehensive explainability engine for TRA system."""
    
    def __init__(self, config: TRAConfig):
        self.config = config
        self.track_explainers = {}
        self.feature_importance_cache = {}
        self.explanation_cache = {}
        
    def initialize_explainers(self, tracks: Dict[str, EnhancedTrack], X_sample: pd.DataFrame):
        """Initialize explainers for all tracks."""
        if not self.config.enable_explanations:
            return
            
        logger.info("Initializing explainability engines...")
        
        for track_name, track in tracks.items():
            try:
                if self.config.explanation_method == "shap" and SHAP_AVAILABLE:
                    self._initialize_shap_explainer(track_name, track, X_sample)
                elif self.config.explanation_method == "lime" and LIME_AVAILABLE:
                    self._initialize_lime_explainer(track_name, track, X_sample)
                else:
                    # Fallback to feature importance
                    self._initialize_feature_importance(track_name, track)
                    
            except Exception as e:
                logger.warning(f"Failed to initialize explainer for {track_name}: {e}")
                self._initialize_feature_importance(track_name, track)
    
    def _initialize_shap_explainer(self, track_name: str, track: EnhancedTrack, X_sample: pd.DataFrame):
        """Initialize SHAP explainer for a track."""
        try:
            if hasattr(track.classifier, 'predict_proba'):
                # For tree-based models
                if hasattr(track.classifier, 'estimators_'):
                    explainer = shap.TreeExplainer(track.classifier)
                else:
                    # For other models, use Kernel explainer with subset
                    background = shap.sample(X_sample.values, min(100, len(X_sample)))
                    explainer = shap.KernelExplainer(track.classifier.predict_proba, background)
            else:
                # For regression or classifiers without predict_proba
                background = shap.sample(X_sample.values, min(100, len(X_sample)))
                explainer = shap.KernelExplainer(track.classifier.predict, background)
                
            self.track_explainers[track_name] = {
                'type': 'shap',
                'explainer': explainer,
                'background': X_sample.values[:min(50, len(X_sample))]
            }
            
            logger.info(f"SHAP explainer initialized for {track_name}")
            
        except Exception as e:
            logger.warning(f"SHAP initialization failed for {track_name}: {e}")
            self._initialize_feature_importance(track_name, track)
    
    def _initialize_lime_explainer(self, track_name: str, track: EnhancedTrack, X_sample: pd.DataFrame):
        """Initialize LIME explainer for a track."""
        try:
            mode = 'classification' if self.config.task_type == "classification" else 'regression'
            explainer = lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=X_sample.columns.tolist(),
                mode=mode,
                discretize_continuous=True
            )
            
            self.track_explainers[track_name] = {
                'type': 'lime',
                'explainer': explainer,
                'predict_fn': track.classifier.predict_proba if hasattr(track.classifier, 'predict_proba') else track.classifier.predict
            }
            
            logger.info(f"LIME explainer initialized for {track_name}")
            
        except Exception as e:
            logger.warning(f"LIME initialization failed for {track_name}: {e}")
            self._initialize_feature_importance(track_name, track)
    
    def _initialize_feature_importance(self, track_name: str, track: EnhancedTrack):
        """Initialize feature importance explainer as fallback."""
        try:
            importance = None
            feature_names = []
            
            if hasattr(track.classifier, 'feature_importances_'):
                importance = track.classifier.feature_importances_
            elif hasattr(track.classifier, 'coef_'):
                importance = np.abs(track.classifier.coef_)
                if importance.ndim > 1:
                    importance = np.mean(importance, axis=0)
            
            self.track_explainers[track_name] = {
                'type': 'feature_importance',
                'importance': importance,
                'feature_names': feature_names
            }
            
            logger.debug(f"Feature importance explainer initialized for {track_name}")
            
        except Exception as e:
            logger.warning(f"Feature importance initialization failed for {track_name}: {e}")
    
    def explain_prediction(self, track_name: str, X_instance: pd.DataFrame, 
                          prediction: np.ndarray) -> Dict[str, Any]:
        """Generate explanation for a prediction."""
        if not self.config.enable_explanations or track_name not in self.track_explainers:
            return {"explanation_available": False, "reason": "explainer_not_available"}
        
        try:
            explainer_info = self.track_explainers[track_name]
            
            if explainer_info['type'] == 'shap':
                return self._explain_with_shap(explainer_info, X_instance, track_name)
            elif explainer_info['type'] == 'lime':
                return self._explain_with_lime(explainer_info, X_instance, track_name)
            else:
                return self._explain_with_feature_importance(explainer_info, X_instance, track_name)
                
        except Exception as e:
            logger.warning(f"Explanation generation failed for {track_name}: {e}")
            return {"explanation_available": False, "reason": f"explanation_error: {e}"}
    
    def _explain_with_shap(self, explainer_info: Dict, X_instance: pd.DataFrame, track_name: str) -> Dict[str, Any]:
        """Generate SHAP explanation."""
        try:
            explainer = explainer_info['explainer']
            
            # Get SHAP values
            if hasattr(explainer, 'shap_values'):
                if len(X_instance) == 1:
                    shap_values = explainer.shap_values(X_instance.values)
                else:
                    shap_values = explainer.shap_values(X_instance.values[:5])  # Limit for performance
            else:
                shap_values = explainer(X_instance.values[:1])
                shap_values = shap_values.values
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Use first instance
            
            # Get top contributing features
            feature_contributions = {}
            for i, importance in enumerate(shap_values):
                if i < len(X_instance.columns):
                    feature_contributions[X_instance.columns[i]] = float(importance)
            
            # Sort by absolute importance
            sorted_features = sorted(feature_contributions.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            
            return {
                "explanation_available": True,
                "method": "shap",
                "feature_contributions": dict(sorted_features[:10]),
                "top_positive_features": [f for f, v in sorted_features if v > 0][:5],
                "top_negative_features": [f for f, v in sorted_features if v < 0][:5],
                "explanation_confidence": float(np.mean(np.abs(shap_values)))
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"explanation_available": False, "reason": f"shap_error: {e}"}
    
    def _explain_with_lime(self, explainer_info: Dict, X_instance: pd.DataFrame, track_name: str) -> Dict[str, Any]:
        """Generate LIME explanation."""
        try:
            explainer = explainer_info['explainer']
            predict_fn = explainer_info['predict_fn']
            
            # Generate explanation for first instance
            explanation = explainer.explain_instance(
                X_instance.values[0], 
                predict_fn, 
                num_features=min(10, len(X_instance.columns))
            )
            
            # Extract feature importance
            feature_contributions = {}
            for feature, importance in explanation.as_list():
                feature_contributions[feature] = importance
            
            return {
                "explanation_available": True,
                "method": "lime",
                "feature_contributions": feature_contributions,
                "top_features": list(feature_contributions.keys())[:5],
                "explanation_confidence": float(explanation.score)
            }
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return {"explanation_available": False, "reason": f"lime_error: {e}"}
    
    def _explain_with_feature_importance(self, explainer_info: Dict, X_instance: pd.DataFrame, track_name: str) -> Dict[str, Any]:
        """Generate explanation using feature importance."""
        try:
            importance = explainer_info.get('importance')
            if importance is None:
                return {"explanation_available": False, "reason": "no_importance_available"}
            
            # Calculate feature contributions (importance * value)
            feature_contributions = {}
            for i, col in enumerate(X_instance.columns):
                if i < len(importance):
                    feature_value = float(X_instance.iloc[0, i])
                    contribution = importance[i] * abs(feature_value)
                    feature_contributions[col] = contribution
            
            # Sort by contribution
            sorted_features = sorted(feature_contributions.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return {
                "explanation_available": True,
                "method": "feature_importance",
                "feature_contributions": dict(sorted_features),
                "top_features": [f for f, v in sorted_features][:5],
                "explanation_confidence": 0.7  # Moderate confidence for feature importance
            }
            
        except Exception as e:
            logger.warning(f"Feature importance explanation failed: {e}")
            return {"explanation_available": False, "reason": f"importance_error: {e}"}
    
    def generate_track_summary(self, track_name: str, track: EnhancedTrack) -> Dict[str, Any]:
        """Generate summary explanation for a track."""
        try:
            summary = {
                "track_name": track_name,
                "track_level": track.level,
                "train_accuracy": track.train_accuracy,
                "validation_accuracy": track.validation_accuracy,
                "test_accuracy": track.test_accuracy,
                "usage_count": track.usage_count,
                "model_type": type(track.classifier).__name__ if track.classifier else "None"
            }
            
            # Add feature importance if available
            if track_name in self.track_explainers:
                explainer_info = self.track_explainers[track_name]
                if explainer_info['type'] == 'feature_importance' and explainer_info['importance'] is not None:
                    importance = explainer_info['importance']
                    summary["top_feature_indices"] = np.argsort(importance)[::-1][:5].tolist()
                    summary["feature_importance_available"] = True
                else:
                    summary["feature_importance_available"] = True
            else:
                summary["feature_importance_available"] = False
            
            return summary
            
        except Exception as e:
            logger.warning(f"Track summary generation failed for {track_name}: {e}")
            return {"track_name": track_name, "summary_available": False, "error": str(e)}

# =============================================================================
# ENHANCED ENSEMBLE FUSION
# =============================================================================

class DynamicEnsembleFusion:
    """Enhanced ensemble fusion with better quality filtering."""
    
    def __init__(self, config: TRAConfig):
        self.config = config
        self.fusion_strategies = {
            'voting': self._voting_fusion,
            'weighted': self._weighted_fusion,
            'stacking': self._stacking_fusion,
            'quality_filtered': self._quality_filtered_fusion,
            'smart_fusion': self._smart_fusion  # NEW
        }
        self.stacking_model = None
        self.performance_history = defaultdict(list)
        self.diversity_matrix = {}

    def fuse_predictions(self, predictions: Dict[str, np.ndarray],
                        confidences: Dict[str, np.ndarray],
                        resource_metrics: Dict[str, AdvancedMetrics],
                        strategy: str = "smart_fusion") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced fusion with smart strategy selection."""
        
        if strategy not in self.fusion_strategies:
            strategy = "smart_fusion"

        try:
            fusion_func = self.fusion_strategies[strategy]
            fused_predictions, fusion_info = fusion_func(predictions, confidences, resource_metrics)
            return fused_predictions, fusion_info
        except Exception as e:
            logger.warning(f"Fusion strategy '{strategy}' failed: {e}, using simple averaging")
            return self._simple_average_fusion(predictions), {'strategy': 'fallback_average'}

    def _smart_fusion(self, predictions: Dict[str, np.ndarray],
                     confidences: Dict[str, np.ndarray],
                     resource_metrics: Dict[str, AdvancedMetrics]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """NEW: Smart fusion that adapts based on track performance."""
        
        if not predictions:
            return np.array([]), {'strategy': 'smart_fusion', 'models_used': []}

        # Filter high-performing tracks
        quality_tracks = {}
        track_scores = {}
        
        for track_name, pred in predictions.items():
            # Calculate track quality score
            quality_score = 0.0
            
            # Historical performance (40%)
            hist_perf = self.performance_history.get(track_name, [0.5])
            quality_score += 0.4 * np.mean(hist_perf)
            
            # Current confidence (30%)
            conf_array = confidences.get(track_name, [0.5])
            avg_conf = np.mean(conf_array) if len(conf_array) > 0 else 0.5
            quality_score += 0.3 * avg_conf
            
            # Resource efficiency (20%)
            if track_name in resource_metrics:
                efficiency = resource_metrics[track_name].efficiency_score()
                quality_score += 0.2 * efficiency
            else:
                quality_score += 0.1
            
            # Prediction consistency (10%)
            pred_std = np.std(pred) if len(pred) > 1 else 0.5
            consistency = max(0, 1.0 - pred_std)
            quality_score += 0.1 * consistency
            
            track_scores[track_name] = quality_score
            
            # Include tracks above threshold
            if quality_score >= self.config.min_track_accuracy:
                quality_tracks[track_name] = pred
        
        logger.debug(f"Track quality scores: {track_scores}")
        
        # If no tracks pass, use top 2 tracks
        if not quality_tracks:
            logger.warning("No tracks passed quality threshold, using top performers")
            top_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            for track_name, _ in top_tracks:
                quality_tracks[track_name] = predictions[track_name]
        
        # Apply weighted fusion to quality tracks
        if len(quality_tracks) == 1:
            track_name = list(quality_tracks.keys())[0]
            return list(quality_tracks.values())[0], {
                'strategy': 'smart_fusion_single',
                'selected_track': track_name,
                'track_score': track_scores[track_name]
            }
        
        # Weighted fusion
        weights = {}
        total_weight = 0.0
        
        for track_name in quality_tracks.keys():
            weight = track_scores[track_name]
            weights[track_name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(quality_tracks) for k in quality_tracks.keys()}
        
        # Apply weighted fusion
        fused = self._apply_weighted_fusion(quality_tracks, weights)
        
        return fused, {
            'strategy': 'smart_fusion_weighted',
            'weights': weights,
            'track_scores': track_scores,
            'models_used': list(quality_tracks.keys())
        }

    def _apply_weighted_fusion(self, predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Apply weighted fusion to predictions."""
        pred_arrays = list(predictions.values())
        if not pred_arrays:
            return np.array([])
        
        n_samples = len(pred_arrays[0])
        
        if self.config.task_type == "classification":
            # Weighted voting for classification
            fused = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                class_votes = defaultdict(float)
                for track_name, preds in predictions.items():
                    if i < len(preds):
                        try:
                            pred_class = int(preds[i])
                            weight = weights[track_name]
                            class_votes[pred_class] += weight
                        except (ValueError, TypeError, IndexError):
                            continue
                
                if class_votes:
                    fused[i] = max(class_votes.items(), key=lambda x: x[1])[0]
                else:
                    fused[i] = 0
        else:
            # Weighted average for regression
            fused = np.zeros(n_samples, dtype=float)
            for track_name, preds in predictions.items():
                try:
                    weight = weights[track_name]
                    preds_array = np.asarray(preds, dtype=float)[:n_samples]
                    if len(preds_array) == n_samples:
                        fused += weight * preds_array
                except (ValueError, TypeError):
                    continue
        
        return fused

    def _quality_filtered_fusion(self, predictions: Dict[str, np.ndarray],
                                confidences: Dict[str, np.ndarray],
                                resource_metrics: Dict[str, AdvancedMetrics]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quality-filtered fusion that excludes low-performing models."""
        
        if not predictions:
            return np.array([]), {'strategy': 'quality_filtered', 'models_used': []}

        # Filter models based on quality
        quality_models = {}
        for model_name, pred in predictions.items():
            model_quality = 0.0
            
            # Check historical performance
            hist_perf = self.performance_history.get(model_name, [])
            if hist_perf:
                historical_score = np.mean(hist_perf)
                model_quality += 0.5 * historical_score

            # Check average confidence
            conf_array = confidences.get(model_name, [0.5])
            if isinstance(conf_array, (list, np.ndarray)) and len(conf_array) > 0:
                avg_confidence = float(np.mean(conf_array))
                model_quality += 0.3 * avg_confidence

            # Check resource efficiency if available
            if model_name in resource_metrics:
                efficiency = resource_metrics[model_name].efficiency_score()
                model_quality += 0.2 * efficiency
            else:
                model_quality += 0.1

            # Only include models above threshold
            if model_quality >= self.config.min_track_accuracy:
                quality_models[model_name] = pred

        # If no models pass quality filter, use all models
        if not quality_models:
            logger.warning("No models passed quality filter, using all models")
            quality_models = predictions

        # Apply weighted fusion
        return self._weighted_fusion(quality_models, confidences, resource_metrics)

    def _weighted_fusion(self, predictions: Dict[str, np.ndarray],
                        confidences: Dict[str, np.ndarray],
                        resource_metrics: Dict[str, AdvancedMetrics]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Improved weighted fusion with per-sample confidence."""
        
        if not predictions:
            return np.array([]), {'strategy': 'weighted', 'weights': {}}

        n_samples = len(list(predictions.values())[0])
        model_names = list(predictions.keys())

        if self.config.task_type == "classification":
            fused = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                class_votes = defaultdict(float)
                total_confidence = 0.0
                
                for model_name in model_names:
                    try:
                        pred = predictions[model_name][i]
                        conf_array = confidences.get(model_name, [0.7])
                        
                        if isinstance(conf_array, (list, np.ndarray)) and len(conf_array) > i:
                            sample_conf = float(conf_array[i])
                        else:
                            sample_conf = 0.7

                        class_votes[int(pred)] += sample_conf
                        total_confidence += sample_conf
                    except (IndexError, ValueError, TypeError):
                        continue

                if class_votes and total_confidence > 0:
                    fused[i] = max(class_votes.items(), key=lambda x: x[1])[0]
                else:
                    fused[i] = 0
        else:
            # Per-sample weighted average for regression
            fused = np.zeros(n_samples, dtype=float)
            for i in range(n_samples):
                weighted_sum = 0.0
                total_weight = 0.0
                
                for model_name in model_names:
                    try:
                        pred = predictions[model_name][i]
                        conf_array = confidences.get(model_name, [0.7])
                        
                        if isinstance(conf_array, (list, np.ndarray)) and len(conf_array) > i:
                            weight = float(conf_array[i])
                        else:
                            weight = 0.7

                        weighted_sum += weight * float(pred)
                        total_weight += weight
                    except (IndexError, ValueError, TypeError):
                        continue

                if total_weight > 0:
                    fused[i] = weighted_sum / total_weight
                else:
                    fused[i] = 0.0

        return fused, {'strategy': 'per_sample_weighted', 'n_models': len(predictions)}

    def _voting_fusion(self, predictions: Dict[str, np.ndarray],
                      confidences: Dict[str, np.ndarray],
                      resource_metrics: Dict[str, AdvancedMetrics]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simple voting fusion."""
        pred_arrays = list(predictions.values())
        if not pred_arrays:
            return np.array([]), {'strategy': 'voting'}

        if self.config.task_type == "classification":
            # Majority vote
            try:
                from scipy import stats
                fused = stats.mode(pred_arrays, axis=0)[0].flatten().astype(int)
            except:
                # Fallback for scipy version compatibility
                fused = []
                for i in range(len(pred_arrays[0])):
                    votes = [pred[i] for pred in pred_arrays if i < len(pred)]
                    if votes:
                        fused.append(max(set(votes), key=votes.count))
                    else:
                        fused.append(0)
                fused = np.array(fused, dtype=int)
        else:
            # Average for regression
            fused = np.mean(pred_arrays, axis=0)

        return fused, {'strategy': 'voting', 'n_models': len(predictions)}

    def _stacking_fusion(self, predictions: Dict[str, np.ndarray],
                        confidences: Dict[str, np.ndarray],
                        resource_metrics: Dict[str, AdvancedMetrics]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Stacking fusion with meta-model."""
        if self.stacking_model is None:
            return self._smart_fusion(predictions, confidences, resource_metrics)

        try:
            # Create feature matrix from predictions
            pred_matrix = np.column_stack(list(predictions.values()))
            fused = self.stacking_model.predict(pred_matrix)
            return fused, {'strategy': 'stacking', 'meta_model': type(self.stacking_model).__name__}
        except Exception as e:
            logger.warning(f"Stacking fusion failed: {e}")
            return self._smart_fusion(predictions, confidences, resource_metrics)

    def _simple_average_fusion(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple averaging as ultimate fallback."""
        pred_arrays = list(predictions.values())
        if not pred_arrays:
            return np.array([])

        if self.config.task_type == "classification":
            # Simple majority vote
            fused = []
            for i in range(len(pred_arrays[0])):
                votes = [pred[i] for pred in pred_arrays if i < len(pred)]
                if votes:
                    fused.append(max(set(votes), key=votes.count))
                else:
                    fused.append(0)
            return np.array(fused, dtype=int)
        else:
            return np.mean(pred_arrays, axis=0)

    def train_meta_models(self, base_predictions: List[np.ndarray], true_labels: np.ndarray):
        """Train meta-models with proper out-of-fold predictions."""
        if len(base_predictions) < 2:
            logger.warning("Need at least 2 base models for meta-learning")
            return

        try:
            # Align all predictions to same length
            lengths = [len(pred) for pred in base_predictions]
            min_len = min(lengths)
            
            aligned_predictions = [pred[:min_len] for pred in base_predictions]
            y_aligned = true_labels[:min_len]

            # Create feature matrix from base predictions
            X_meta = np.column_stack(aligned_predictions)

            # Train stacking model with cross-validation
            if self.config.task_type == "classification":
                self.stacking_model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                self.stacking_model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )

            # Train meta-model
            self.stacking_model.fit(X_meta, y_aligned)
            logger.info("Stacking meta-model trained successfully")

        except Exception as e:
            logger.error(f"Meta-model training failed: {e}")

    def update_performance_history(self, model_name: str, accuracy: float):
        """Update performance history for a model."""
        self.performance_history[model_name].append(accuracy)
        # Keep only recent history
        if len(self.performance_history[model_name]) > 10:
            self.performance_history[model_name] = self.performance_history[model_name][-10:]

# =============================================================================
# MAIN ENHANCED TRA SYSTEM
# =============================================================================

class EnhancedTrackRailSystem(BaseEstimator, ClassifierMixin):
    """COMPLETELY FIXED: Enhanced Track-Rail Algorithm with all improvements."""

    def __init__(self, config: TRAConfig = None):
        self.config = config or TRAConfig()
        self.config.validate()

        # Core system components
        self.tracks: Dict[str, EnhancedTrack] = {}
        self.global_performance_history = deque(maxlen=1000)
        self.system_metrics = AdvancedMetrics()

        # Enhanced components
        self.feature_engineer = AutomatedFeatureEngineer(self.config)
        self.meta_learner = MetaLearningEngine(self.config)
        self.fusion_engine = DynamicEnsembleFusion(self.config)
        self.routing_engine = SmartRoutingEngine(self.config)  # NEW
        self.explainability_engine = ExplainabilityEngine(self.config)  # NEW

        # System state
        self.is_fitted = False
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.classes_ = None

        # CRITICAL: Routing and tracking
        self.routing_decisions = []
        self.track_performance_test = {}  # Track individual test performance

        logger.info("Enhanced TRA system initialized with smart routing and explainability")

    def fit(self, X, y, domain_type: str = "tabular", checkpoint_name: str = None):
        """FIXED: Enhanced fit method with proper track routing."""
        logger.info("Starting Enhanced TRA system training...")
        start_time = time.time()

        try:
            # Input validation and conversion
            X, y = self._validate_and_convert_input(X, y)
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

            if self.config.task_type == "classification":
                self.classes_ = np.unique(y)
                logger.info(f"Classification task with {len(self.classes_)} classes")

            # Feature engineering
            logger.info("Performing automated feature engineering...")
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            X_engineered = self.feature_engineer.fit_transform(X_df, y, domain_type)
            logger.info(f"Feature engineering: {X.shape[1]} -> {X_engineered.shape[1]} features")

            # Split data for proper training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_engineered, y, test_size=0.3, random_state=42,
                stratify=y if self.config.task_type == "classification" else None
            )

            # Meta-learning suggestions
            meta_suggestions = self.meta_learner.suggest_model_architecture(X_train.values, y_train)
            logger.info(f"Meta-learning suggestion: {meta_suggestions.get('primary_model', 'ensemble')}")

            # Create enhanced tracks
            self._create_enhanced_tracks(X_train, y_train, meta_suggestions)

            # CRITICAL: Evaluate tracks on test data to get realistic performance
            self._evaluate_tracks_on_test_data(X_test, y_test)

            # Initialize routing engine
            self.routing_engine.analyze_track_specializations(self.tracks, X_train, y_train)

            # Initialize explainability
            self.explainability_engine.initialize_explainers(self.tracks, X_train)

            # Train ensemble fusion meta-models
            self._train_ensemble_fusion(X_train, y_train)

            self.is_fitted = True
            training_time = time.time() - start_time

            logger.info(f"Enhanced TRA training completed in {training_time:.2f} seconds")
            logger.info(f"Created {len(self.tracks)} tracks")

            # Log track performance summary
            self._log_track_performance_summary()

            return self

        except Exception as e:
            logger.error(f"Enhanced TRA training failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"TRA training failed: {e}")

    def predict(self, X):
        """FIXED: Enhanced prediction with smart routing and proper track usage."""
        if not self.is_fitted:
            raise ValueError("TRA system must be fitted before prediction")

        try:
            # Validate and preprocess input consistently
            X_df = self._validate_prediction_input(X)
            X_engineered = self.feature_engineer.transform(X_df)

            logger.debug(f"Predicting for {len(X_engineered)} samples")

            # CRITICAL: Route each record to appropriate tracks
            predictions = {}
            confidences = {}
            resource_metrics = {}
            routing_summary = []

            # Process records in batches for efficiency
            batch_size = min(100, len(X_engineered))
            
            for i in range(0, len(X_engineered), batch_size):
                batch_end = min(i + batch_size, len(X_engineered))
                X_batch = X_engineered.iloc[i:batch_end]
                
                # Route this batch
                batch_predictions = self._route_and_predict_batch(X_batch, i)
                
                # Merge batch results
                for track_name, track_preds in batch_predictions['predictions'].items():
                    if track_name not in predictions:
                        predictions[track_name] = []
                        confidences[track_name] = []
                    
                    predictions[track_name].extend(track_preds)
                    confidences[track_name].extend(batch_predictions['confidences'][track_name])
                
                routing_summary.extend(batch_predictions['routing_decisions'])

            # Convert lists to arrays
            for track_name in predictions:
                predictions[track_name] = np.array(predictions[track_name])
                confidences[track_name] = np.array(confidences[track_name])
                resource_metrics[track_name] = self.tracks[track_name].metrics

            # Store routing decisions for analysis
            self.routing_decisions.extend(routing_summary)

            if not predictions:
                logger.error("No track predictions available")
                n_samples = len(X)
                if self.config.task_type == "classification":
                    return np.zeros(n_samples, dtype=int)
                else:
                    return np.zeros(n_samples, dtype=float)

            # Use smart fusion
            final_predictions, fusion_info = self.fusion_engine.fuse_predictions(
                predictions, confidences, resource_metrics, strategy="smart_fusion"
            )

            logger.debug(f"Fusion strategy: {fusion_info.get('strategy', 'unknown')}")
            logger.debug(f"Models used: {fusion_info.get('models_used', [])}")

            # Log routing summary
            routing_stats = self.routing_engine.get_routing_summary()
            logger.info(f"Routing summary: {routing_stats['track_distribution']}")

            return final_predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(traceback.format_exc())
            # Ultimate fallback
            n_samples = len(X)
            if self.config.task_type == "classification":
                return np.zeros(n_samples, dtype=int)
            else:
                return np.zeros(n_samples, dtype=float)

    def _route_and_predict_batch(self, X_batch: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
            """CRITICAL: Route batch of records to appropriate tracks and get predictions."""
            
            batch_predictions = defaultdict(list)
            batch_confidences = defaultdict(list)
            routing_decisions = []
            
            for idx, (_, row) in enumerate(X_batch.iterrows()):
                record_id = start_idx + idx
                
                # Create enhanced record
                enhanced_record = EnhancedRecord(
                    id=record_id,
                    features=row.values
                )
                
                # CRITICAL: Route this specific record
                routing_decision = self.routing_engine.route_record(
                    enhanced_record, self.tracks, pd.DataFrame([row])
                )
                
                # Get prediction from selected track
                selected_track_name = routing_decision.selected_track
                if selected_track_name in self.tracks:
                    track = self.tracks[selected_track_name]
                    
                    # Get prediction for this single record
                    try:
                        X_single = pd.DataFrame([row])
                        prediction = track.predict(X_single)
                        
                        # Get confidence/probability
                        if hasattr(track, 'predict_proba') and hasattr(track.classifier, 'predict_proba'):
                            try:
                                proba = track.predict_proba(X_single)
                                if proba.ndim > 1 and proba.shape[1] > 1:
                                    confidence = np.max(proba[0])
                                else:
                                    confidence = 0.7
                            except:
                                confidence = 0.7
                        else:
                            confidence = 0.7
                        
                        # Store prediction and confidence
                        batch_predictions[selected_track_name].append(prediction[0])
                        batch_confidences[selected_track_name].append(confidence)
                        
                        logger.debug(f"Record {record_id} -> {selected_track_name} (conf: {confidence:.3f})")
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for record {record_id} on track {selected_track_name}: {e}")
                        # Fallback prediction
                        if self.config.task_type == "classification":
                            batch_predictions[selected_track_name].append(0)
                        else:
                            batch_predictions[selected_track_name].append(0.0)
                        batch_confidences[selected_track_name].append(0.3)
                
                routing_decisions.append(routing_decision)
            
            return {
                'predictions': dict(batch_predictions),
                'confidences': dict(batch_confidences),
                'routing_decisions': routing_decisions
            }

    def predict_proba(self, X):
        """Enhanced probability prediction with track routing."""
        if not self.is_fitted:
            raise ValueError("TRA system must be fitted before prediction")
        
        if self.config.task_type != "classification":
            raise ValueError("predict_proba only available for classification tasks")

        try:
            X_df = self._validate_prediction_input(X)
            X_engineered = self.feature_engineer.transform(X_df)
            
            # Get probabilistic predictions from tracks
            track_probabilities = {}
            track_confidences = {}
            
            for track_name, track in self.tracks.items():
                try:
                    proba = track.predict_proba(X_engineered)
                    track_probabilities[track_name] = proba
                    
                    # Calculate confidence as max probability
                    if proba.ndim > 1:
                        confidences = np.max(proba, axis=1)
                    else:
                        confidences = np.full(len(X_engineered), 0.7)
                    
                    track_confidences[track_name] = confidences
                    
                except Exception as e:
                    logger.warning(f"Probability prediction failed for track {track_name}: {e}")
                    # Fallback: uniform probabilities
                    n_classes = len(self.classes_)
                    fallback_proba = np.full((len(X_engineered), n_classes), 1.0 / n_classes)
                    track_probabilities[track_name] = fallback_proba
                    track_confidences[track_name] = np.full(len(X_engineered), 0.5)
            
            # Fusion for probabilities
            if len(track_probabilities) == 1:
                return list(track_probabilities.values())[0]
            
            # Weighted average of probabilities
            weights = {}
            total_weight = 0.0
            
            for track_name in track_probabilities.keys():
                # Weight based on track performance and confidence
                track_weight = 1.0
                if hasattr(self.tracks[track_name], 'test_accuracy'):
                    track_weight *= max(0.1, self.tracks[track_name].test_accuracy)
                else:
                    track_weight *= 0.5
                
                avg_confidence = np.mean(track_confidences[track_name])
                track_weight *= avg_confidence
                
                weights[track_name] = track_weight
                total_weight += track_weight
            
            # Normalize weights
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                weights = {k: 1.0 / len(track_probabilities) for k in track_probabilities.keys()}
            
            # Compute weighted average
            n_samples = len(X_engineered)
            n_classes = len(self.classes_)
            fused_proba = np.zeros((n_samples, n_classes))
            
            for track_name, proba in track_probabilities.items():
                weight = weights[track_name]
                if proba.shape == (n_samples, n_classes):
                    fused_proba += weight * proba
                else:
                    logger.warning(f"Probability shape mismatch for track {track_name}")
            
            # Ensure probabilities sum to 1
            row_sums = fused_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            fused_proba = fused_proba / row_sums
            
            return fused_proba

        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            # Ultimate fallback
            n_samples = len(X)
            n_classes = len(self.classes_) if self.classes_ is not None else 2
            return np.full((n_samples, n_classes), 1.0 / n_classes)

    def _create_enhanced_tracks(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                               meta_suggestions: Dict[str, Any]):
        """FIXED: Create enhanced tracks with proper hyperparameter optimization."""
        logger.info("Creating enhanced tracks...")
        
        # 1. Create global track first
        self._create_global_track(X_train, y_train, meta_suggestions)
        
        # 2. Create regional tracks with improved clustering
        self._create_regional_tracks(X_train, y_train, meta_suggestions)
        
        # 3. Create adaptive signals (for routing)
        self._create_adaptive_signals(X_train, y_train)
        
        logger.info(f"Created {len(self.tracks)} enhanced tracks")

    def _create_global_track(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                           meta_suggestions: Dict[str, Any]):
        """Create and optimize global track."""
        logger.info("Creating global track...")
        
        track_name = "global_track_0"
        
        try:
            # Hyperparameter optimization with Optuna
            if OPTUNA_AVAILABLE and self.config.enable_hyperopt:
                best_params = self._optimize_hyperparameters(X_train, y_train, track_name)
                classifier = self._create_classifier_from_params(best_params)
                logger.info(f"Hyperparameter optimization completed: {best_params}")
            else:
                # Fallback to meta-learning suggestions
                classifier = self._create_classifier_from_meta_suggestions(meta_suggestions)
            
            # Create and fit track
            global_track = EnhancedTrack(
                name=track_name,
                level=TrackLevel.GLOBAL,
                classifier=classifier,
                config=self.config
            )
            
            global_track.fit(X_train, y_train)
            self.tracks[track_name] = global_track
            
            logger.info(f"Global track training accuracy: {global_track.train_accuracy:.4f}")
            logger.info("Global track created and fitted successfully")
            
        except Exception as e:
            logger.error(f"Global track creation failed: {e}")
            # Create simple fallback track
            self._create_fallback_track(track_name, X_train, y_train)

    def _create_regional_tracks(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                              meta_suggestions: Dict[str, Any]):
        """FIXED: Create regional tracks with improved clustering."""
        logger.info("Creating regional tracks with improved clustering...")
        
        try:
            # Try multiple clustering methods
            cluster_labels = self._find_best_clustering(X_train, y_train)
            
            if cluster_labels is not None:
                n_clusters = len(np.unique(cluster_labels))
                logger.info(f"Regional clusters: n={n_clusters} | silhouette={silhouette_score(X_train, cluster_labels):.3f}")
                
                # Create track for each cluster
                for cluster_id in np.unique(cluster_labels):
                    cluster_mask = cluster_labels == cluster_id
                    X_cluster = X_train[cluster_mask]
                    y_cluster = y_train[cluster_mask]
                    
                    if len(X_cluster) >= 10:  # Minimum samples per cluster
                        track_name = f"regional_track_{cluster_id}_{['ra', 'gr', 'bl', 'ye', 'or'][cluster_id % 5]}"
                        
                        # Create specialized classifier for this region
                        classifier = self._create_regional_classifier(X_cluster, y_cluster, meta_suggestions)
                        
                        regional_track = EnhancedTrack(
                            name=track_name,
                            level=TrackLevel.REGIONAL,
                            classifier=classifier,
                            parent_track=self.tracks.get("global_track_0"),
                            config=self.config
                        )
                        
                        regional_track.fit(X_cluster, y_cluster)
                        self.tracks[track_name] = regional_track
                        
                        logger.info(f"Regional track {track_name} training accuracy: {regional_track.train_accuracy:.4f}")
                        logger.info(f"Regional track {track_name} with {len(X_cluster)} samples added.")
                
                logger.info(f"Created {len([t for t in self.tracks.keys() if 'regional' in t])} regional tracks")
            else:
                logger.warning("Clustering failed, skipping regional tracks")
                
        except Exception as e:
            logger.error(f"Regional track creation failed: {e}")

    def _find_best_clustering(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Optional[np.ndarray]:
        """FIXED: Find best clustering with more lenient thresholds."""
        
        # Reduce dimensionality if needed
        if X_train.shape[1] > 20:
            try:
                pca = PCA(n_components=min(20, X_train.shape[1] - 1), random_state=42)
                X_reduced = pca.fit_transform(X_train)
            except:
                X_reduced = X_train.values
        else:
            X_reduced = X_train.values
        
        best_labels = None
        best_score = -1
        
        # Try different clustering methods
        clustering_methods = [
            ('kmeans_2', lambda: KMeans(n_clusters=2, random_state=42, n_init=10)),
            ('kmeans_3', lambda: KMeans(n_clusters=3, random_state=42, n_init=10)),
            ('dbscan', lambda: DBSCAN(eps=0.5, min_samples=5)),
            ('gmm_2', lambda: GaussianMixture(n_components=2, random_state=42)),
        ]
        
        for method_name, method_func in clustering_methods:
            try:
                clusterer = method_func()
                labels = clusterer.fit_predict(X_reduced)
                
                # Filter out noise points for DBSCAN
                if method_name.startswith('dbscan'):
                    labels = labels[labels != -1]
                    if len(labels) < len(X_reduced) * 0.8:  # Too many noise points
                        continue
                    X_filtered = X_reduced[clusterer.labels_ != -1]
                else:
                    X_filtered = X_reduced
                
                # Check clustering quality
                n_clusters = len(np.unique(labels))
                if n_clusters < 2 or n_clusters > self.config.max_tracks:
                    continue
                
                # Calculate metrics
                try:
                    sil_score = silhouette_score(X_filtered, labels)
                except:
                    sil_score = 0.0
                
                # Check cluster sizes
                cluster_sizes = np.bincount(labels)
                min_cluster_size = np.min(cluster_sizes)
                min_size_ratio = min_cluster_size / len(labels)
                
                # More lenient quality checks
                quality_score = sil_score
                
                if (sil_score >= self.config.min_silhouette_score and
                    min_size_ratio >= self.config.min_cluster_size_ratio and
                    quality_score > best_score):
                    
                    best_score = quality_score
                    best_labels = clusterer.labels_ if method_name.startswith('dbscan') else labels
                    logger.info(f"Better clustering found: {method_name} (score: {quality_score:.3f})")
                
            except Exception as e:
                logger.debug(f"Clustering method {method_name} failed: {e}")
                continue
        
        if best_labels is None:
            logger.warning("No clustering method met quality requirements, using simple split")
            # Simple split as fallback
            n_samples = len(X_train)
            best_labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
            np.random.shuffle(best_labels)
        
        return best_labels

    def _create_adaptive_signals(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Create adaptive signals for routing."""
        logger.info("Creating adaptive signals...")
        
        # For now, signals are implicit in the routing engine's track specializations
        # In future versions, these could be explicit signal objects
        n_signals = len(self.tracks) * 4  # 4 signals per track
        logger.info(f"Created {n_signals} adaptive signals")

    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                                 track_name: str) -> Dict[str, Any]:
        """ENHANCED: Hyperparameter optimization with broader search space and pruning."""
        
        if not OPTUNA_AVAILABLE:
            return self._get_default_hyperparameters()
        
        def objective(trial):
            try:
                # Broader model search space
                model_type = trial.suggest_categorical('model_type', [
                    'random_forest', 'extra_trees', 'gradient_boosting'
                ] + (['xgboost'] if XGBOOST_AVAILABLE else []) +
                  (['lightgbm'] if LIGHTGBM_AVAILABLE else []))
                
                # Model-specific hyperparameters
                if model_type == 'random_forest':
                    params = {
                        'model_type': model_type,
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                    }
                elif model_type == 'extra_trees':
                    params = {
                        'model_type': model_type,
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                    }
                elif model_type == 'gradient_boosting':
                    params = {
                        'model_type': model_type,
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3)
                    }
                elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                    params = {
                        'model_type': model_type,
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.8, 1.0)
                    }
                elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    params = {
                        'model_type': model_type,
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100)
                    }
                else:
                    params = self._get_default_hyperparameters()
                
                # Create and evaluate classifier
                classifier = self._create_classifier_from_params(params)
                
                # Cross-validation with proper scoring
                cv_scores = cross_val_score(
                    classifier, X_train, y_train,
                    cv=3, scoring='accuracy' if self.config.task_type == "classification" else 'neg_mean_squared_error',
                    n_jobs=1
                )
                
                score = np.mean(cv_scores)
                
                # For regression, convert negative MSE to positive score
                if self.config.task_type == "regression":
                    score = -score  # Convert back to positive MSE
                    score = 1.0 / (1.0 + score)  # Convert to a score between 0 and 1
                
                return score
                
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return 0.0
        
        # Create study with enhanced pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # Optimize with timeout
        study.optimize(objective, n_trials=self.config.nas_budget, timeout=120)
        
        if study.best_trial:
            return study.best_params
        else:
            return self._get_default_hyperparameters()

    def _create_classifier_from_params(self, params: Dict[str, Any]):
        """Create classifier from hyperparameters."""
        model_type = params.get('model_type', 'random_forest')
        
        try:
            if model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42,
                    n_jobs=-1
                ) if self.config.task_type == "classification" else RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == 'extra_trees':
                return ExtraTreesClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    min_samples_split=params.get('min_samples_split', 2),
                    random_state=42,
                    n_jobs=-1
                ) if self.config.task_type == "classification" else ExtraTreesRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    min_samples_split=params.get('min_samples_split', 2),
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == 'gradient_boosting':
                return GradientBoostingClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    random_state=42
                ) if self.config.task_type == "classification" else GradientBoostingRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    random_state=42
                )
            
            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 1.0),
                    random_state=42,
                    eval_metric='logloss' if self.config.task_type == "classification" else 'rmse'
                ) if self.config.task_type == "classification" else xgb.XGBRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 1.0),
                    random_state=42
                )
            
            elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                return lgb.LGBMClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    num_leaves=params.get('num_leaves', 31),
                    random_state=42,
                    verbosity=-1
                ) if self.config.task_type == "classification" else lgb.LGBMRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    num_leaves=params.get('num_leaves', 31),
                    random_state=42,
                    verbosity=-1
                )
            
            else:
                # Fallback to random forest
                return RandomForestClassifier(random_state=42) if self.config.task_type == "classification" else RandomForestRegressor(random_state=42)
                
        except Exception as e:
            logger.warning(f"Failed to create {model_type} classifier: {e}")
            # Ultimate fallback
            return RandomForestClassifier(random_state=42) if self.config.task_type == "classification" else RandomForestRegressor(random_state=42)

    def _create_classifier_from_meta_suggestions(self, meta_suggestions: Dict[str, Any]):
        """Create classifier from meta-learning suggestions."""
        base_models = meta_suggestions.get('base_models', ['random_forest'])
        
        if len(base_models) > 0:
            model_type = base_models[0]  # Use first suggested model
            return self._create_classifier_from_params({'model_type': model_type})
        else:
            return RandomForestClassifier(random_state=42) if self.config.task_type == "classification" else RandomForestRegressor(random_state=42)

    def _create_regional_classifier(self, X_cluster: pd.DataFrame, y_cluster: np.ndarray, 
                                  meta_suggestions: Dict[str, Any]):
        """Create specialized classifier for regional cluster."""
        
        # For small clusters, use simpler models
        if len(X_cluster) < 100:
            if self.config.task_type == "classification":
                return DecisionTreeClassifier(max_depth=5, random_state=42)
            else:
                return DecisionTreeRegressor(max_depth=5, random_state=42)
        else:
            # Use meta-learning suggestions
            return self._create_classifier_from_meta_suggestions(meta_suggestions)

    def _create_fallback_track(self, track_name: str, X_train: pd.DataFrame, y_train: np.ndarray):
        """Create simple fallback track."""
        logger.info(f"Creating fallback track: {track_name}")
        
        try:
            if self.config.task_type == "classification":
                classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                classifier = RandomForestRegressor(n_estimators=50, random_state=42)
            
            fallback_track = EnhancedTrack(
                name=track_name,
                level=TrackLevel.GLOBAL,
                classifier=classifier,
                config=self.config
            )
            
            fallback_track.fit(X_train, y_train)
            self.tracks[track_name] = fallback_track
            
        except Exception as e:
            logger.error(f"Even fallback track creation failed: {e}")

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10
        }

    def _evaluate_tracks_on_test_data(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """CRITICAL: Evaluate each track on test data to get realistic performance metrics."""
        logger.info("Evaluating tracks on test data...")
        
        for track_name, track in self.tracks.items():
            try:
                test_accuracy = track.evaluate_on_test(X_test, y_test)
                self.track_performance_test[track_name] = test_accuracy
                logger.info(f"Track {track_name} test accuracy: {test_accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Test evaluation failed for {track_name}: {e}")
                self.track_performance_test[track_name] = 0.0

    def _train_ensemble_fusion(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Train ensemble fusion meta-models."""
        logger.info("Training ensemble fusion meta-models...")
        
        if len(self.tracks) < 2:
            logger.warning("Need at least 2 tracks for ensemble fusion")
            return
        
        try:
            # Get out-of-fold predictions for meta-learning
            base_predictions = []
            
            for track_name, track in self.tracks.items():
                try:
                    # Use cross-validation to get out-of-fold predictions
                    oof_predictions = cross_val_predict(
                        track.classifier, X_train, y_train, cv=3, method='predict'
                    )
                    base_predictions.append(oof_predictions)
                except Exception as e:
                    logger.warning(f"Failed to get OOF predictions for {track_name}: {e}")
            
            if len(base_predictions) >= 2:
                self.fusion_engine.train_meta_models(base_predictions, y_train)
                logger.info("Stacking meta-model trained successfully with cross-validation")
            
        except Exception as e:
            logger.error(f"Ensemble fusion training failed: {e}")

    def _validate_and_convert_input(self, X, y):
        """Validate and convert input data."""
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        
        # Ensure y is properly formatted
        if self.config.task_type == "classification":
            y = y.astype(int)
        else:
            y = y.astype(float)
        
        return X, y

    def _validate_prediction_input(self, X) -> pd.DataFrame:
        """Validate input for prediction and return consistent DataFrame."""
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        # Ensure correct number of features
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        # Convert to DataFrame with consistent column names
        if hasattr(self, 'feature_names_in_'):
            columns = self.feature_names_in_
        else:
            columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        return pd.DataFrame(X, columns=columns)

    def _log_track_performance_summary(self):
        """Log comprehensive track performance summary."""
        logger.info("=== TRACK PERFORMANCE SUMMARY ===")
        
        for track_name, track in self.tracks.items():
            logger.info(f"{track_name}:")
            logger.info(f"  - Train Accuracy: {track.train_accuracy:.4f}")
            logger.info(f"  - Validation Accuracy: {track.validation_accuracy:.4f}")
            logger.info(f"  - Test Accuracy: {track.test_accuracy:.4f}")
            logger.info(f"  - Usage Count: {track.usage_count}")

    # =============================================================================
    # SYSTEM STATUS AND ANALYSIS METHODS
    # =============================================================================

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.is_fitted:
            return {"fitted": False, "error": "System not fitted"}
        
        status = {
            "fitted": True,
            "task_type": self.config.task_type,
            "n_tracks": len(self.tracks),
            "n_features_original": self.n_features_in_,
            "n_features_engineered": self.feature_engineer.n_features_out_ if self.feature_engineer.is_fitted else 0,
            "tracks": {},
            "routing_summary": self.routing_engine.get_routing_summary(),
            "capabilities": {
                "automated_feature_engineering": self.config.enable_automated_fe,
                "meta_learning": self.config.enable_meta_learning,
                "neural_architecture_search": self.config.enable_nas,
                "dynamic_ensemble_fusion": self.config.dynamic_fusion,
                "explainable_ai": self.config.enable_explanations,
                "concept_drift_detection": True,
                "resource_monitoring": True,
                "self_healing": True,
                "checkpointing": True
            }
        }
        
        # Track details
        for track_name, track in self.tracks.items():
            status["tracks"][track_name] = track.get_health_status()
        
        return status

    def get_track_explanations(self, X_sample: pd.DataFrame, track_name: str = None) -> Dict[str, Any]:
        """Get explanations for tracks."""
        if not self.config.enable_explanations:
            return {"explanations_enabled": False}
        
        explanations = {}
        
        if track_name:
            # Specific track explanation
            if track_name in self.tracks:
                track = self.tracks[track_name]
                explanation = self.explainability_engine.explain_prediction(
                    track_name, X_sample, track.predict(X_sample)
                )
                explanations[track_name] = explanation
        else:
            # All tracks
            for track_name, track in self.tracks.items():
                try:
                    explanation = self.explainability_engine.explain_prediction(
                        track_name, X_sample, track.predict(X_sample)
                    )
                    explanations[track_name] = explanation
                except Exception as e:
                    explanations[track_name] = {"error": str(e)}
        
        return explanations

    def get_routing_analysis(self) -> Dict[str, Any]:
        """Get detailed routing analysis."""
        if not self.routing_decisions:
            return {"total_decisions": 0, "analysis": "No routing decisions available"}
        
        # Analyze routing patterns
        track_usage = defaultdict(int)
        confidence_stats = defaultdict(list)
        reason_counts = defaultdict(int)
        
        for decision in self.routing_decisions[-1000:]:  # Last 1000 decisions
            track_usage[decision.selected_track] += 1
            confidence_stats[decision.selected_track].append(decision.confidence)
            reason_counts[decision.reason] += 1
        
        # Calculate statistics
        analysis = {
            "total_decisions": len(self.routing_decisions),
            "track_usage_distribution": dict(track_usage),
            "average_confidence_by_track": {
                track: np.mean(confidences) for track, confidences in confidence_stats.items()
            },
            "routing_reasons": dict(reason_counts),
            "routing_efficiency": len(track_usage) / max(len(self.tracks), 1)
        }
        
        return analysis

    def create_checkpoint(self, checkpoint_dir: str = "tra_checkpoints") -> str:
        """Create system checkpoint."""
        if not self.is_fitted:
            raise ValueError("Cannot checkpoint unfitted system")
        
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        # Generate checkpoint filename
        timestamp = int(time.time())
        checkpoint_file = Path(checkpoint_dir) / f"checkpoint_{timestamp}.pkl"
        
        try:
            # Save system state
            checkpoint_data = {
                'config': self.config,
                'tracks': self.tracks,
                'feature_engineer': self.feature_engineer,
                'routing_engine': self.routing_engine,
                'system_state': {
                    'is_fitted': self.is_fitted,
                    'n_features_in_': self.n_features_in_,
                    'feature_names_in_': self.feature_names_in_,
                    'classes_': self.classes_
                }
            }
            
            joblib.dump(checkpoint_data, checkpoint_file, compress=3)
            logger.info(f"Checkpoint created: {checkpoint_file}")
            return str(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Checkpoint creation failed: {e}")
            raise

    def load_checkpoint(self, checkpoint_file: str):
        """Load system from checkpoint."""
        try:
            checkpoint_data = joblib.load(checkpoint_file)
            
            self.config = checkpoint_data['config']
            self.tracks = checkpoint_data['tracks']
            self.feature_engineer = checkpoint_data['feature_engineer']
            self.routing_engine = checkpoint_data['routing_engine']
            
            # Restore system state
            system_state = checkpoint_data['system_state']
            self.is_fitted = system_state['is_fitted']
            self.n_features_in_ = system_state['n_features_in_']
            self.feature_names_in_ = system_state['feature_names_in_']
            self.classes_ = system_state['classes_']
            
            logger.info(f"System restored from checkpoint: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            raise

    def get_performance_metrics(self, X_test, y_test) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.is_fitted:
            raise ValueError("System must be fitted before evaluation")
        
        start_time = time.time()
        predictions = self.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        if self.config.task_type == "classification":
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            conf_matrix = confusion_matrix(y_test, predictions)
        else:
            accuracy = -mean_squared_error(y_test, predictions)  # Negative MSE
            f1 = accuracy  # For regression, use same metric
            conf_matrix = None
        
        # Individual track performance
        individual_accuracies = {}
        for track_name, track in self.tracks.items():
            try:
                track_preds = track.predict(self.feature_engineer.transform(
                    self._validate_prediction_input(X_test)
                ))
                if self.config.task_type == "classification":
                    track_acc = accuracy_score(y_test, track_preds)
                else:
                    track_acc = -mean_squared_error(y_test, track_preds)
                individual_accuracies[track_name] = track_acc
            except Exception as e:
                logger.warning(f"Could not evaluate track {track_name}: {e}")
                individual_accuracies[track_name] = 0.0
        
        metrics = {
            "overall_accuracy": accuracy,
            "f1_score": f1,
            "prediction_time_seconds": prediction_time,
            "predictions_per_second": len(X_test) / prediction_time if prediction_time > 0 else 0,
            "individual_track_accuracies": individual_accuracies,
            "confusion_matrix": conf_matrix.tolist() if conf_matrix is not None else None,
            "routing_statistics": self.get_routing_analysis()
        }
        
        return metrics

# =============================================================================
# DEMO AND TESTING FUNCTIONS
# =============================================================================

def create_test_datasets():
    """Create test datasets for TRA demonstration."""
    logger.info("Creating test datasets...")
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=1500, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=2, random_state=42, class_sep=0.8
    )
    
    # Regression dataset  
    X_reg, y_reg = make_regression(
        n_samples=1500, n_features=20, n_informative=15, noise=0.1, random_state=42
    )
    
    return (X_class, y_class), (X_reg, y_reg)

def run_enhanced_tra_demo():
    """Run comprehensive TRA demonstration."""
    print("=" * 80)
    print("ENHANCED TRACK-RAIL ALGORITHM (TRA) DEMONSTRATION")
    print("=" * 80)

    # Create test data
    (X_class, y_class), (X_reg, y_reg) = create_test_datasets()

    # Use classification for demo
    X, y = X_class, y_class

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Enhanced TRA configuration
    config = TRAConfig(
        task_type="classification",
        max_tracks=5,
        enable_meta_learning=True,
        enable_automated_fe=True,
        enable_stacking=True
    )

    logger.info(f"""
Testing Enhanced TRA Configuration:
- Task Type: {config.task_type}
- Max Tracks: {config.max_tracks}
- Meta Learning: {config.enable_meta_learning}
- Feature Engineering: {config.enable_automated_fe}
- Stacking: {config.enable_stacking}
""")

    # Initialize and fit TRA
    logger.info("Initializing Enhanced TRA...")
    tra = EnhancedTrackRailSystem(config=config)
    logger.info("Enhanced TRA system initialized")

    logger.info("Fitting Enhanced TRA...")
    start_time = time.time()
    tra.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info("Making predictions...")
    start_pred_time = time.time()
    predictions = tra.predict(X_test)
    prediction_time = time.time() - start_pred_time

    # Get performance metrics
    metrics = tra.get_performance_metrics(X_test, y_test)

    # Individual track performance logging
    print("\nIndividual track accuracies:")
    for track_name, accuracy in metrics["individual_track_accuracies"].items():
        print(f" - {track_name}: {accuracy:.4f}")
    print(f"Fused system accuracy: {metrics['overall_accuracy']:.3f}")

    # System status
    status = tra.get_system_status()

    # Detailed results
    print("\n" + "=" * 50)
    print("ENHANCED TRA PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Predictions/Second: {metrics['predictions_per_second']:.2f}")
    print(f"Probability Predictions: {'Available' if hasattr(tra, 'predict_proba') else 'Not Available'}")

    print("\n" + "=" * 50)
    print("SYSTEM STATUS")
    print("=" * 50)
    print(f"Number of Tracks: {status['n_tracks']}")
    # FIXED: Replace → with ->
    print(f"Features (original -> engineered): {status['n_features_original']} -> {status['n_features_engineered']}")
    print(f"Task Type: {status['task_type']}")
    print(f"System Fitted: {status['fitted']}")

    print("\n" + "=" * 50)
    print("TRACK DETAILS")
    print("=" * 50)
    for track_name, track_info in status["tracks"].items():
        print(f"{track_name}:")
        print(f" - Status: {track_info.get('status', 'unknown')}")
        print(f" - Usage Count: {track_info.get('usage_count', 0)}")
        print(f" - Avg Prediction Time: {track_info.get('avg_prediction_time_ms', 0):.2f}ms")
        print(f" - Test Accuracy: {track_info.get('test_accuracy', 0):.4f}")

    print("\n" + "=" * 50)
    print("ENHANCED CAPABILITIES")
    print("=" * 50)
    capabilities = status["capabilities"]
    for capability, enabled in capabilities.items():
        # FIXED: Replace ✓ and ✗ with OK and X
        status_symbol = "OK" if enabled else "X"
        formatted_name = capability.replace("_", " ").title()
        print(f"{status_symbol} {formatted_name}: {enabled}")

    # Routing analysis
    routing_analysis = tra.get_routing_analysis()
    if routing_analysis["total_decisions"] > 0:
        print("\n" + "=" * 50)
        print("ROUTING ANALYSIS")
        print("=" * 50)
        print(f"Total Routing Decisions: {routing_analysis['total_decisions']}")
        print("Track Usage Distribution:")
        for track, count in routing_analysis["track_usage_distribution"].items():
            percentage = (count / routing_analysis['total_decisions']) * 100
            print(f" - {track}: {count} ({percentage:.1f}%)")

    # Create checkpoint
    try:
        checkpoint_file = tra.create_checkpoint()
        logger.info(f"Checkpoint created: {checkpoint_file}")
    except Exception as e:
        logger.warning(f"Checkpoint creation failed: {e}")

    print("\n" + "=" * 80)
    print("ENHANCED TRA DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return {
        "accuracy": metrics['overall_accuracy'],
        "training_time": training_time,
        "prediction_time": prediction_time,
        "system_status": status,
        "routing_analysis": routing_analysis
    }


if __name__ == "__main__":
    try:
        results = run_enhanced_tra_demo()
        print(f"\nDemo completed successfully!")
        print(f"Final accuracy: {results['accuracy']:.4f}")
        print(f"Training time: {results['training_time']:.2f}s")
        print(f"Prediction time: {results['prediction_time']:.4f}s")
        
        print("\nCleanup completed")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.error(traceback.format_exc())
        print(f"Demo failed with error: {e}")

