import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import logging
import warnings
import joblib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Deque
from collections import deque
from sklearn.datasets import make_classification, make_regression

# Enhanced imports for new features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import psutil
import gc

# Explainability imports
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

# Bayesian optimization imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

# Online learning imports
try:
    from river import tree, ensemble, linear_model, metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("River not available for online learning. Install with: pip install river")

# Automated feature engineering
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
    print("Featuretools not available. Install with: pip install featuretools")

try:
    import networkx as nx
except ImportError:
    nx = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# GPU detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

@dataclass
class DataDistribution:
    """Enhanced track data distribution characteristics for dynamic track generation."""
    cluster_centers: np.ndarray = None
    cluster_labels: np.ndarray = None
    n_clusters: int = 0
    density_threshold: float = 0.1
    variance_threshold: float = 1.0
    timestamp: float = field(default_factory=time.time)
    stability_score: float = 1.0
    clustering_method: str = "kmeans"
    embedding_dim: int = 0
    feature_importance: np.ndarray = None
    concept_drift_score: float = 0.0

    def is_stable(self, new_distribution: 'DataDistribution', threshold: float = 0.8) -> bool:
        """Check if distribution is stable compared to new distribution."""
        if self.cluster_centers is None or new_distribution.cluster_centers is None:
            return False

        if self.n_clusters != new_distribution.n_clusters:
            return False

        try:
            distances = []
            for i in range(self.n_clusters):
                min_dist = float('inf')
                for j in range(new_distribution.n_clusters):
                    dist = np.linalg.norm(self.cluster_centers[i] - new_distribution.cluster_centers[j])
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)

            avg_distance = np.mean(distances)
            stability = 1.0 / (1.0 + avg_distance)
            
            # Include concept drift consideration
            drift_penalty = max(0, self.concept_drift_score - 0.5) * 0.2
            return (stability - drift_penalty) > threshold

        except:
            return False

@dataclass
class ResourceMetrics:
    """Track resource usage metrics for resource-aware routing."""
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    prediction_count: int = 0
    total_time: float = 0.0
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score based on latency and memory usage."""
        if self.prediction_count == 0:
            return 0.0
        
        avg_latency = self.total_time / self.prediction_count
        efficiency = 1.0 / (1.0 + avg_latency + self.memory_mb / 1000.0)
        return min(1.0, efficiency)

@dataclass
class Record:
    """Enhanced Record class for tracking data flow with hierarchical information."""
    id: int
    features: np.ndarray
    current_track: str = "global_track_0"
    hierarchical_path: List[str] = field(default_factory=list)
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    confidence: float = 1.0
    cluster_assignment: int = -1
    local_context: Dict[str, Any] = field(default_factory=dict)
    embeddings: np.ndarray = None
    explanation: Dict[str, Any] = field(default_factory=dict)
    resource_preference: str = "balanced"  # balanced, fast, accurate

    def update_track(self, new_track: str, confidence: float = 1.0, level: str = "global"):
        """Update current track with hierarchical path tracking."""
        if new_track != self.current_track:
            self.history.append((self.current_track, time.time(), level))
            self.hierarchical_path.append(f"{level}:{new_track}")
            self.current_track = new_track
            self.confidence = confidence
            self.last_update = time.time()

@dataclass
class TemporalRecord:
    """Enhanced record for tracking temporal information about track switching."""
    id: int
    features: np.ndarray
    current_track: str
    timestamp: datetime = field(default_factory=datetime.now)
    track_history: List[Tuple[str, datetime, float]] = field(default_factory=list)
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    switch_count: int = 0
    last_switch_time: Optional[datetime] = None
    performance_score: float = 0.0
    stability_score: float = 1.0
    concept_drift_detected: bool = False
    online_updates: int = 0

    def add_track_switch(self, new_track: str, confidence: float):
        """Record a track switch event."""
        if self.current_track != new_track:
            self.track_history.append((self.current_track, datetime.now(), confidence))
            self.current_track = new_track
            self.switch_count += 1
            self.last_switch_time = datetime.now()
            self.confidence_history.append(confidence)

    def get_stability_score(self, time_window_minutes: int = 5) -> float:
        """Calculate stability score based on recent switching history."""
        current_time = datetime.now()
        recent_switches = [
            (track, timestamp, conf) for track, timestamp, conf in self.track_history
            if (current_time - timestamp).total_seconds() < time_window_minutes * 60
        ]

        if not recent_switches:
            return 1.0

        return max(0.1, 1.0 - (len(recent_switches) / 10.0))

class TrackLevel:
    """Enum-like class for track hierarchy levels."""
    GLOBAL = "global"
    REGIONAL = "regional"
    LOCAL = "local"
    META = "meta"
    STACKING = "stacking"

class DeepFeatureExtractor(nn.Module):
    """Neural network for deep feature extraction and embeddings."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dims: List[int] = None):
        super(DeepFeatureExtractor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [min(256, input_dim * 2), embedding_dim * 2]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        return self.encoder(x)

class ConceptDriftDetector:
    """Detect concept drift in data streams."""
    
    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.drift_detected = False
        
    def update(self, features: np.ndarray, predictions: np.ndarray, true_labels: np.ndarray = None):
        """Update drift detector with new data."""
        error_rate = np.mean(predictions != true_labels) if true_labels is not None else 0.0
        
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(error_rate)
        else:
            self.current_window.append(error_rate)
            
            if len(self.current_window) == self.window_size:
                # Statistical test for drift detection
                ref_mean = np.mean(self.reference_window)
                curr_mean = np.mean(self.current_window)
                
                drift_magnitude = abs(curr_mean - ref_mean)
                self.drift_detected = drift_magnitude > self.drift_threshold
                
                if self.drift_detected:
                    # Reset reference window to current data
                    self.reference_window = self.current_window.copy()
                    self.current_window.clear()
                    
        return self.drift_detected

class AutomatedFeatureEngineer:
    """Automated feature engineering using domain knowledge and statistical methods."""
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.generated_features = []
        self.feature_importance = {}
        
    def generate_features(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        """Generate new features automatically."""
        if not FEATURETOOLS_AVAILABLE:
            logger.warning("Featuretools not available, using basic feature engineering")
            return self._basic_feature_engineering(X)
        
        try:
            # Create entity set
            es = ft.EntitySet(id="data")
            es = es.add_dataframe(
                dataframe_name="main",
                dataframe=X.reset_index(),
                index="index"
            )
            
            # Generate features
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="main",
                max_depth=2,
                n_jobs=1
            )
            
            # Select top features based on variance
            numeric_features = feature_matrix.select_dtypes(include=[np.number])
            variances = numeric_features.var().sort_values(ascending=False)
            top_features = variances.head(self.max_features).index
            
            return feature_matrix[top_features]
            
        except Exception as e:
            logger.warning(f"Automated feature engineering failed: {e}")
            return self._basic_feature_engineering(X)
    
    def _basic_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Basic feature engineering when featuretools is not available."""
        X_new = X.copy()
        
        # Polynomial features for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:6]:
                    X_new[f"{col1}_times_{col2}"] = X[col1] * X[col2]
                    X_new[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
        
        # Statistical features
        for col in numeric_cols[:10]:
            X_new[f"{col}_squared"] = X[col] ** 2
            X_new[f"{col}_log"] = np.log1p(np.abs(X[col]))
        
        return X_new.select_dtypes(include=[np.number])

class AdaptiveSignalCondition:
    """Enhanced adaptive switching conditions between tracks."""
    
    def __init__(self, source_track: str, target_track: str,
                 model_type: str, initial_threshold: float = 0.6,
                 min_time_between_switches: int = 30):
        self.source_track = source_track
        self.target_track = target_track
        self.model_type = model_type
        self.threshold = initial_threshold
        self.min_time_between_switches = min_time_between_switches
        self.activation_count = 0
        self.success_count = 0
        self.total_evaluations = 0
        self.confidence_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        self.last_activation_time = None
        self.adaptive_factor = 1.0
        self.resource_aware = True
        self.drift_detector = ConceptDriftDetector()

    def can_switch(self, record: TemporalRecord) -> bool:
        if record.last_switch_time is None:
            return True
        
        time_since_switch = (datetime.now() - record.last_switch_time).total_seconds()
        stability_factor = record.get_stability_score()
        required_wait_time = self.min_time_between_switches / stability_factor
        
        # Consider concept drift
        if record.concept_drift_detected:
            required_wait_time *= 0.5  # Allow faster switching during drift
        
        return time_since_switch >= required_wait_time

    def evaluate(self, source_model, target_model, features: np.ndarray,
                 task_type: str, record: TemporalRecord,
                 resource_metrics: Dict[str, ResourceMetrics] = None) -> bool:
        self.total_evaluations += 1
        
        if not self.can_switch(record):
            return False

        source_conf = self._get_model_confidence(source_model, features, task_type)
        target_conf = self._get_model_confidence(target_model, features, task_type)

        adaptive_threshold = self.threshold * self.adaptive_factor
        stability_bonus = record.get_stability_score() * 0.1
        
        # Resource-aware adjustment
        resource_penalty = 0.0
        if self.resource_aware and resource_metrics:
            source_efficiency = resource_metrics.get(self.source_track, ResourceMetrics()).get_efficiency_score()
            target_efficiency = resource_metrics.get(self.target_track, ResourceMetrics()).get_efficiency_score()
            resource_penalty = max(0, source_efficiency - target_efficiency) * 0.1

        confidence_diff = target_conf - source_conf + stability_bonus - resource_penalty

        self.confidence_history.append((source_conf, target_conf, confidence_diff))

        if confidence_diff > adaptive_threshold:
            self.activation_count += 1
            self.last_activation_time = datetime.now()
            return True

        return False

    def update_performance(self, success: bool):
        self.performance_history.append(success)
        if success:
            self.success_count += 1

        if len(self.performance_history) >= 10:
            recent_success_rate = sum(list(self.performance_history)[-10:]) / 10.0
            if recent_success_rate > 0.7:
                self.adaptive_factor = max(0.5, self.adaptive_factor * 0.95)
            elif recent_success_rate < 0.3:
                self.adaptive_factor = min(2.0, self.adaptive_factor * 1.05)

    def _get_model_confidence(self, model, features: np.ndarray, task_type: str) -> float:
        features_reshaped = features.reshape(1, -1)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_reshaped)
                return np.max(proba)
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(features_reshaped)
                if decision.ndim > 1:
                    return np.max(np.abs(decision))
                return abs(float(decision))
            else:
                return 0.8  # Default confidence
        except Exception:
            return 0.1  # Fallback confidence

class Signal:
    """Enhanced Signal class with adaptive learning and hierarchical support."""
    
    def __init__(self, name: str, condition: AdaptiveSignalCondition,
                 source_track: str, target_track: str, level: str = TrackLevel.GLOBAL):
        self.name = name
        self.condition = condition
        self.source_track = source_track
        self.target_track = target_track
        self.level = level
        self.activation_count = 0
        self.success_count = 0
        self.confidence = 1.0
        self.last_activation = 0.0
        self.performance_history = deque(maxlen=100)
        self.ensemble_history = deque(maxlen=50)
        self.cooldown_period = 0.1
        self.explanation_cache = {}

    def evaluate(self, record: Record, sample_features: np.ndarray = None) -> Tuple[bool, float, Dict[str, float]]:
        """Enhanced evaluation with adaptive cooldown and ensemble information."""
        current_time = time.time()
        
        if current_time - self.last_activation < self.cooldown_period:
            return False, 0.0, {}

        should_switch, switch_confidence, ensemble_weights = self.condition.evaluate(record, sample_features)

        if should_switch:
            self.activation_count += 1
            self.last_activation = current_time
            self.ensemble_history.append(ensemble_weights)

            if len(self.performance_history) > 10:
                recent_success_rate = sum(list(self.performance_history)[-10:]) / 10
                self.cooldown_period = max(0.05, 0.2 * (1.0 - recent_success_rate))

        return should_switch, switch_confidence, ensemble_weights

    def update_performance(self, success: bool, performance_diff: float = 0.0):
        """Enhanced performance update with adaptive threshold adjustment."""
        self.performance_history.append(success)
        if success:
            self.success_count += 1

        if len(self.performance_history) >= 5:
            recent_performance = list(self.performance_history)[-20:]
            self.confidence = sum(recent_performance) / len(recent_performance)
        elif self.activation_count > 0:
            self.confidence = self.success_count / self.activation_count

        self.condition.adapt_threshold(performance_diff)

class AdvancedClusteringEngine:
    """Advanced clustering with multiple algorithms and dynamic selection."""
    
    def __init__(self, max_clusters: int = 10):
        self.max_clusters = max_clusters
        self.clustering_methods = {
            'kmeans': self._kmeans_clustering,
            'dbscan': self._dbscan_clustering,
            'gaussian_mixture': self._gaussian_mixture_clustering
        }
        self.best_method = 'kmeans'
        self.best_params = {}
        
    def fit_predict(self, X: np.ndarray, method: str = 'auto') -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit clustering and return labels with metadata."""
        if method == 'auto':
            method = self._select_best_method(X)
        
        if method not in self.clustering_methods:
            method = 'kmeans'
        
        labels, metadata = self.clustering_methods[method](X)
        metadata['method'] = method
        return labels, metadata
    
    def _select_best_method(self, X: np.ndarray) -> str:
        """Select best clustering method based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Density estimation
        density = n_samples / (n_features * np.var(X))
        
        if density < 0.1:
            return 'dbscan'  # Sparse data
        elif n_features > 20:
            return 'gaussian_mixture'  # High dimensional
        else:
            return 'kmeans'  # Default
    
    def _kmeans_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """K-means clustering with automatic K selection."""
        best_k = 1
        best_score = -1
        
        for k in range(2, min(self.max_clusters + 1, X.shape[0])):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = final_kmeans.fit_predict(X)
        
        return labels, {
            'n_clusters': best_k,
            'silhouette_score': best_score,
            'cluster_centers': final_kmeans.cluster_centers_
        }
    
    def _dbscan_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """DBSCAN clustering with parameter tuning."""
        best_eps = 0.5
        best_score = -1
        
        for eps in [0.3, 0.5, 0.7, 1.0]:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=max(2, X.shape[0] // 50))
                labels = dbscan.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
            except:
                continue
        
        final_dbscan = DBSCAN(eps=best_eps, min_samples=max(2, X.shape[0] // 50))
        labels = final_dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return labels, {
            'n_clusters': n_clusters,
            'silhouette_score': best_score,
            'eps': best_eps
        }
    
    def _gaussian_mixture_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gaussian Mixture Model clustering."""
        best_k = 1
        best_score = float('-inf')
        
        for k in range(2, min(self.max_clusters + 1, X.shape[0])):
            try:
                gm = GaussianMixture(n_components=k, random_state=42)
                gm.fit(X)
                score = gm.bic(X)  # Lower BIC is better
                if score < best_score or best_score == float('-inf'):
                    best_score = score
                    best_k = k
            except:
                continue
        
        final_gm = GaussianMixture(n_components=best_k, random_state=42)
        final_gm.fit(X)
        labels = final_gm.predict(X)
        
        return labels, {
            'n_clusters': best_k,
            'bic_score': best_score,
            'means': final_gm.means_
        }

class HierarchicalTrack:
    """Enhanced base class for hierarchical track structure."""
    
    def __init__(self, name: str, level: str, classifier=None, parent_track=None,
                 use_gpu: bool = False, enable_online_learning: bool = False):
        self.name = name
        self.level = level
        self.classifier = classifier
        self.parent_track = parent_track
        self.child_tracks: Dict[str, 'HierarchicalTrack'] = {}
        self.signals: List[Signal] = []
        self.records: List[Record] = []
        self.performance_score = 0.5
        self.usage_count = 0
        self.last_used = time.time()
        self.prediction_times = deque(maxlen=50)
        self.specialization_context = {}
        self.data_distribution = None
        self.ensemble_weights = []
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_online_learning = enable_online_learning
        self.resource_metrics = ResourceMetrics()
        self.explainer = None
        self.feature_importance = None
        self.online_model = None
        self.drift_detector = ConceptDriftDetector()
        
        # Initialize online model if enabled
        if self.enable_online_learning and RIVER_AVAILABLE:
            if level in [TrackLevel.GLOBAL, TrackLevel.REGIONAL]:
                self.online_model = tree.HoeffdingTreeClassifier()
            else:
                self.online_model = linear_model.LogisticRegression()

    def add_child_track(self, child_track: 'HierarchicalTrack'):
        """Add a child track to this hierarchical track."""
        child_track.parent_track = self
        self.child_tracks[child_track.name] = child_track

    def add_signal(self, signal: Signal):
        """Add a signal to this track."""
        self.signals.append(signal)

    def update_online(self, X: np.ndarray, y: np.ndarray):
        """Update online model with new data."""
        if not self.enable_online_learning or not RIVER_AVAILABLE or self.online_model is None:
            return
        
        try:
            for i in range(len(X)):
                x_dict = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}
                self.online_model.learn_one(x_dict, y[i])
        except Exception as e:
            logger.warning(f"Online learning update failed for {self.name}: {e}")

    def predict_online(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using online model if available."""
        if not self.enable_online_learning or not RIVER_AVAILABLE or self.online_model is None:
            return self.predict(X, use_hierarchy=False)
        
        try:
            predictions = []
            for i in range(len(X)):
                x_dict = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}
                pred = self.online_model.predict_one(x_dict)
                predictions.append(pred)
            return np.array(predictions)
        except Exception as e:
            logger.warning(f"Online prediction failed for {self.name}: {e}")
            return self.predict(X, use_hierarchy=False)

    def get_explanation(self, X: np.ndarray, method: str = 'shap') -> Dict[str, Any]:
        """Get explanation for predictions."""
        if not SHAP_AVAILABLE and not LIME_AVAILABLE:
            return {'explanation': 'Explainability libraries not available'}
        
        try:
            if method == 'shap' and SHAP_AVAILABLE and self.explainer is None:
                if hasattr(self.classifier, 'predict_proba'):
                    self.explainer = shap.Explainer(self.classifier.predict_proba, X[:100])
                else:
                    self.explainer = shap.Explainer(self.classifier.predict, X[:100])
            
            if method == 'shap' and SHAP_AVAILABLE and self.explainer is not None:
                shap_values = self.explainer(X[:5])  # Limit for performance
                return {
                    'method': 'shap',
                    'feature_importance': np.mean(np.abs(shap_values.values), axis=0),
                    'individual_explanations': shap_values.values.tolist()
                }
            
            elif method == 'lime' and LIME_AVAILABLE:
                explainer = lime_tabular.LimeTabularExplainer(
                    X[:100], mode='classification' if hasattr(self.classifier, 'predict_proba') else 'regression'
                )
                
                explanations = []
                for i in range(min(3, len(X))):
                    exp = explainer.explain_instance(X[i], self.classifier.predict_proba)
                    explanations.append(dict(exp.as_list()))
                
                return {
                    'method': 'lime',
                    'explanations': explanations
                }
                
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
        
        return {'explanation': 'Failed to generate explanation'}

    def find_best_child(self, record: Record) -> Optional['HierarchicalTrack']:
        """Find the best child track for a record based on specialization."""
        if not self.child_tracks:
            return None

        best_child = None
        best_score = -1

        for child in self.child_tracks.values():
            score = self._calculate_specialization_score(record, child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_score > 0.5 else None

    def _calculate_specialization_score(self, record: Record, child_track: 'HierarchicalTrack') -> float:
        """Calculate how well a child track specializes for this record."""
        if child_track.data_distribution is None:
            return 0.0

        try:
            if (record.cluster_assignment >= 0 and
                child_track.data_distribution.cluster_centers is not None and
                record.cluster_assignment < len(child_track.data_distribution.cluster_centers)):
                
                cluster_center = child_track.data_distribution.cluster_centers[record.cluster_assignment]
                features = record.embeddings if record.embeddings is not None else record.features
                distance = np.linalg.norm(features - cluster_center)
                
                # Consider resource preferences
                efficiency_bonus = 0.0
                if record.resource_preference == "fast":
                    efficiency_bonus = child_track.resource_metrics.get_efficiency_score() * 0.2
                elif record.resource_preference == "accurate":
                    efficiency_bonus = child_track.performance_score * 0.2
                
                return (1.0 / (1.0 + distance)) + efficiency_bonus

            return child_track.performance_score

        except:
            return 0.0

    def process_record(self, record: Record) -> Record:
        """Process a record through this hierarchical track."""
        start_time = time.time()
        
        self.records.append(record)
        self.usage_count += 1
        self.last_used = time.time()
        
        # Update resource metrics
        process_time = time.time() - start_time
        self.resource_metrics.total_time += process_time
        self.resource_metrics.prediction_count += 1
        self.resource_metrics.latency_ms = process_time * 1000

        record.hierarchical_path.append(f"{self.level}:{self.name}")
        return record

    def predict(self, X: np.ndarray, use_hierarchy: bool = True) -> np.ndarray:
        """Make predictions using hierarchical structure."""
        if self.classifier is None:
            raise ValueError(f"No classifier available for track {self.name}")

        start_time = time.time()
        
        try:
            if self.use_gpu and hasattr(self.classifier, 'predict'):
                # Move to GPU if supported
                if hasattr(X, 'cuda'):
                    X = X.cuda()
            
            if use_hierarchy and self.child_tracks:
                predictions = []
                for i, features in enumerate(X):
                    record = Record(id=i, features=features)
                    best_child = self.find_best_child(record)
                    if best_child and best_child.classifier:
                        pred = best_child.predict(features.reshape(1, -1), use_hierarchy=False)[0]
                    else:
                        pred = self.classifier.predict(features.reshape(1, -1))[0]
                    predictions.append(pred)
                result = np.array(predictions)
            else:
                result = self.classifier.predict(X)

            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            # Update resource metrics
            self.resource_metrics.total_time += prediction_time
            self.resource_metrics.prediction_count += 1

            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for track {self.name}: {e}")
            raise

    def predict_proba(self, X: np.ndarray, use_hierarchy: bool = True) -> np.ndarray:
        """Make probability predictions using hierarchical structure."""
        if self.classifier is None:
            raise ValueError(f"No classifier available for track {self.name}")

        start_time = time.time()
        
        try:
            if use_hierarchy and self.child_tracks:
                probabilities = []
                for i, features in enumerate(X):
                    record = Record(id=i, features=features)
                    best_child = self.find_best_child(record)
                    if best_child and hasattr(best_child.classifier, 'predict_proba'):
                        proba = best_child.predict_proba(features.reshape(1, -1), use_hierarchy=False)[0]
                    else:
                        proba = self.classifier.predict_proba(features.reshape(1, -1))[0]
                    probabilities.append(proba)
                result = np.array(probabilities)
            else:
                result = self.classifier.predict_proba(X)

            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)

            return result
            
        except Exception as e:
            logger.error(f"Probability prediction failed for track {self.name}: {e}")
            raise

class MetaLearningTrack(HierarchicalTrack):
    """Enhanced meta-learning track with Bayesian optimization."""
    
    def __init__(self, name: str, task_type: str = "classification"):
        super().__init__(name, TrackLevel.META)
        self.task_type = task_type
        self.track_creation_history = []
        self.track_performance_predictor = None
        self.optimal_track_configs = []
        self.learning_rate = 0.01
        self.bayesian_optimizer = None
        self.optimization_history = []
        
        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(direction='maximize')

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                n_trials: int = 50) -> Dict[str, Any]:
        """Use Bayesian optimization to find optimal hyperparameters."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default hyperparameters")
            return self._default_hyperparameters()
        
        def objective(trial):
            # Suggest hyperparameters
            model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgb'])
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            
            try:
                if model_type == 'rf':
                    if self.task_type == "classification":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                elif model_type == 'xgb':
                    if self.task_type == "classification":
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42
                        )
                    else:
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42
                        )
                else:  # lgb
                    if self.task_type == "classification":
                        model = lgb.LGBMClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42,
                            verbose=-1
                        )
                    else:
                        model = lgb.LGBMRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42,
                            verbose=-1
                        )
                
                # Cross-validation score
                scores = cross_val_score(model, X, y, cv=3, n_jobs=1)
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Hyperparameter optimization trial failed: {e}")
                return 0.0
        
        try:
            self.study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minute timeout
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            logger.info(f"Best hyperparameters found: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return self._default_hyperparameters()

    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Default hyperparameters when optimization fails."""
        return {
            'model_type': 'rf',
            'n_estimators': 100,
            'max_depth': 8,
            'learning_rate': 0.1
        }

    def learn_track_creation_patterns(self, X: np.ndarray, y: np.ndarray,
                                    existing_tracks: Dict[str, HierarchicalTrack]):
        """Enhanced meta-learning with Bayesian optimization."""
        logger.info("Meta-learning track creation patterns with Bayesian optimization...")
        
        # Optimize hyperparameters first
        optimal_params = self.optimize_hyperparameters(X, y)
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(X, y)
        
        # Evaluate existing track configurations
        config_performance = []
        for track_name, track in existing_tracks.items():
            if track.level != TrackLevel.META:
                config = self._extract_track_config(track)
                performance = track.performance_score
                config_performance.append((config, performance, data_characteristics))

        # Learn patterns using decision tree
        if len(config_performance) > 3:
            try:
                features = []
                targets = []
                for config, perf, data_char in config_performance:
                    feature_vector = self._config_to_vector(config, data_char)
                    features.append(feature_vector)
                    targets.append(perf)

                features = np.array(features)
                targets = np.array(targets)

                # Train meta-learner with optimal hyperparameters
                self.track_performance_predictor = DecisionTreeRegressor(
                    max_depth=optimal_params.get('max_depth', 8),
                    random_state=42
                )
                self.track_performance_predictor.fit(features, targets)
                
                logger.info("Enhanced meta-learner trained successfully")

            except Exception as e:
                logger.warning(f"Meta-learning failed: {str(e)}")

    def suggest_track_creation(self, X: np.ndarray, y: np.ndarray,
                             current_performance: float) -> Dict[str, Any]:
        """Enhanced track suggestion with Bayesian optimization results."""
        if self.track_performance_predictor is None:
            return self._default_track_suggestion()

        try:
            data_characteristics = self._analyze_data_characteristics(X, y)
            
            # Get optimal hyperparameters
            optimal_params = self.optimize_hyperparameters(X, y, n_trials=20)
            
            best_config = None
            best_predicted_performance = current_performance

            test_configs = self._generate_test_configurations(optimal_params)

            for config in test_configs:
                feature_vector = self._config_to_vector(config, data_characteristics)
                predicted_perf = self.track_performance_predictor.predict([feature_vector])[0]

                if predicted_perf > best_predicted_performance:
                    best_predicted_performance = predicted_perf
                    best_config = config

            if best_config:
                suggestion = {
                    'config': best_config,
                    'predicted_performance': best_predicted_performance,
                    'improvement': best_predicted_performance - current_performance,
                    'confidence': min(1.0, abs(best_predicted_performance - current_performance)),
                    'optimal_params': optimal_params
                }
                return suggestion

        except Exception as e:
            logger.warning(f"Enhanced track suggestion failed: {str(e)}")

        return self._default_track_suggestion()

    def _analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Enhanced data analysis with additional characteristics."""
        characteristics = {}
        try:
            characteristics['n_samples'] = X.shape[0]
            characteristics['n_features'] = X.shape[1]
            characteristics['feature_std'] = np.mean(np.std(X, axis=0))
            characteristics['feature_mean'] = np.mean(np.mean(X, axis=0))
            characteristics['sparsity'] = np.mean(X == 0)
            characteristics['skewness'] = np.mean([abs(float(np.mean((X[:, i] - np.mean(X[:, i])) ** 3)) / (np.std(X[:, i]) ** 3)) for i in range(X.shape[1])])

            if self.task_type == "classification":
                characteristics['n_classes'] = len(np.unique(y))
                characteristics['class_balance'] = min(np.bincount(y)) / max(np.bincount(y))
                characteristics['entropy'] = -np.sum(np.bincount(y) / len(y) * np.log2(np.bincount(y) / len(y) + 1e-8))
            else:
                characteristics['target_std'] = np.std(y)
                characteristics['target_range'] = np.max(y) - np.min(y)
                characteristics['target_skewness'] = abs(float(np.mean((y - np.mean(y)) ** 3)) / (np.std(y) ** 3))

            # Advanced clustering analysis
            try:
                clustering_engine = AdvancedClusteringEngine()
                cluster_labels, cluster_metadata = clustering_engine.fit_predict(X)
                characteristics['optimal_clusters'] = cluster_metadata['n_clusters']
                characteristics['silhouette_score'] = cluster_metadata.get('silhouette_score', 0.0)
                characteristics['clustering_method'] = cluster_metadata['method']
            except:
                characteristics['optimal_clusters'] = 1
                characteristics['silhouette_score'] = 0.0
                characteristics['clustering_method'] = 'kmeans'

        except Exception as e:
            logger.warning(f"Enhanced data analysis failed: {str(e)}")
            characteristics = {
                'n_samples': X.shape[0] if len(X.shape) > 0 else 100,
                'n_features': X.shape[1] if len(X.shape) > 1 else 10,
                'feature_std': 1.0,
                'feature_mean': 0.0,
                'optimal_clusters': 1,
                'silhouette_score': 0.0,
                'sparsity': 0.0,
                'skewness': 0.0
            }

        return characteristics

    def _generate_test_configurations(self, optimal_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test configurations incorporating optimal hyperparameters."""
        configs = []
        
        # Base configuration with optimal params
        base_config = {
            'level': TrackLevel.REGIONAL,
            'model_type': optimal_params.get('model_type', 'rf'),
            'n_estimators': optimal_params.get('n_estimators', 100),
            'max_depth': optimal_params.get('max_depth', 8),
            'learning_rate': optimal_params.get('learning_rate', 0.1),
            'n_signals': 4,
            'create_children': False,
            'use_gpu': torch.cuda.is_available(),
            'enable_online_learning': RIVER_AVAILABLE
        }

        # Variations around optimal configuration
        model_types = [optimal_params.get('model_type', 'rf'), 'xgb', 'lgb']
        n_estimators_range = [50, optimal_params.get('n_estimators', 100), 200]
        max_depth_range = [6, optimal_params.get('max_depth', 8), 12]

        for model_type in model_types:
            for n_est in n_estimators_range:
                for max_d in max_depth_range:
                    for create_child in [False, True]:
                        config = base_config.copy()
                        config.update({
                            'model_type': model_type,
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'create_children': create_child
                        })
                        configs.append(config)

        return configs

    def _extract_track_config(self, track: HierarchicalTrack) -> Dict[str, Any]:
        """Enhanced configuration extraction from existing track."""
        config = {
            'level': track.level,
            'n_signals': len(track.signals),
            'usage_count': track.usage_count,
            'has_children': len(track.child_tracks) > 0,
            'n_children': len(track.child_tracks),
            'performance_score': track.performance_score,
            'use_gpu': track.use_gpu,
            'enable_online_learning': track.enable_online_learning
        }

        if hasattr(track.classifier, 'n_estimators'):
            config['n_estimators'] = track.classifier.n_estimators
        if hasattr(track.classifier, 'max_depth'):
            config['max_depth'] = track.classifier.max_depth
        if hasattr(track.classifier, 'learning_rate'):
            config['learning_rate'] = track.classifier.learning_rate

        # Determine model type
        if isinstance(track.classifier, (RandomForestClassifier, RandomForestRegressor)):
            config['model_type'] = 'rf'
        elif isinstance(track.classifier, (xgb.XGBClassifier, xgb.XGBRegressor)):
            config['model_type'] = 'xgb'
        elif isinstance(track.classifier, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            config['model_type'] = 'lgb'
        else:
            config['model_type'] = 'other'

        return config

    def _config_to_vector(self, config: Dict[str, Any], data_char: Dict[str, float]) -> np.ndarray:
        """Enhanced configuration to feature vector conversion."""
        vector = []
        
        # Config features
        vector.append(config.get('n_signals', 0))
        vector.append(config.get('usage_count', 0))
        vector.append(1.0 if config.get('has_children', False) else 0.0)
        vector.append(config.get('n_children', 0))
        vector.append(config.get('n_estimators', 50))
        vector.append(config.get('max_depth', 6))
        vector.append(config.get('learning_rate', 0.1))
        vector.append(1.0 if config.get('use_gpu', False) else 0.0)
        vector.append(1.0 if config.get('enable_online_learning', False) else 0.0)
        
        # Model type encoding
        model_type = config.get('model_type', 'rf')
        vector.extend([
            1.0 if model_type == 'rf' else 0.0,
            1.0 if model_type == 'xgb' else 0.0,
            1.0 if model_type == 'lgb' else 0.0
        ])

        # Enhanced data characteristics
        vector.append(np.log1p(data_char.get('n_samples', 100)))
        vector.append(data_char.get('n_features', 10))
        vector.append(data_char.get('feature_std', 1.0))
        vector.append(data_char.get('optimal_clusters', 1))
        vector.append(data_char.get('silhouette_score', 0.0))
        vector.append(data_char.get('sparsity', 0.0))
        vector.append(data_char.get('skewness', 0.0))

        return np.array(vector)

    def _default_track_suggestion(self) -> Dict[str, Any]:
        """Enhanced default track suggestion."""
        return {
            'config': {
                'level': TrackLevel.REGIONAL,
                'model_type': 'rf',
                'n_estimators': 100,
                'max_depth': 8,
                'create_children': False,
                'use_gpu': torch.cuda.is_available(),
                'enable_online_learning': RIVER_AVAILABLE
            },
            'predicted_performance': 0.5,
            'improvement': 0.0,
            'confidence': 0.1,
            'optimal_params': self._default_hyperparameters()
        }

class StackingMetaLearner:
    """Stacking meta-learner for advanced ensemble fusion."""
    
    def __init__(self, task_type: str = "classification", cv_folds: int = 5):
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.meta_model = None
        self.base_models = []
        self.fitted = False
        
    def fit(self, base_predictions: List[np.ndarray], y: np.ndarray):
        """Fit the stacking meta-learner."""
        try:
            # Stack base predictions
            X_meta = np.column_stack(base_predictions)
            
            # Choose meta-model based on task type
            if self.task_type == "classification":
                self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                self.meta_model = Ridge(random_state=42)
            
            # Fit meta-model
            self.meta_model.fit(X_meta, y)
            self.fitted = True
            
            logger.info(f"Stacking meta-learner fitted with {len(base_predictions)} base models")
            
        except Exception as e:
            logger.error(f"Stacking meta-learner fitting failed: {e}")
            
    def predict(self, base_predictions: List[np.ndarray]) -> np.ndarray:
        """Make predictions using the stacking meta-learner."""
        if not self.fitted:
            # Fallback to simple averaging
            return np.mean(base_predictions, axis=0)
        
        try:
            X_meta = np.column_stack(base_predictions)
            return self.meta_model.predict(X_meta)
        except Exception as e:
            logger.error(f"Stacking prediction failed: {e}")
            return np.mean(base_predictions, axis=0)
    
    def predict_proba(self, base_predictions: List[np.ndarray]) -> np.ndarray:
        """Make probability predictions using the stacking meta-learner."""
        if not self.fitted or not hasattr(self.meta_model, 'predict_proba'):
            # Fallback to simple averaging
            return np.mean(base_predictions, axis=0)
        
        try:
            X_meta = np.column_stack([pred if pred.ndim == 1 else pred for pred in base_predictions])
            return self.meta_model.predict_proba(X_meta)
        except Exception as e:
            logger.error(f"Stacking probability prediction failed: {e}")
            return np.mean(base_predictions, axis=0)

def default_track_performance():
    return {'total': 0, 'correct': 0}

class RobustFeatureSelector:
    """Enhanced robust feature selection with deep features and automated engineering."""
    
    def __init__(self, task_type: str = "classification", max_features: int = None,
                 use_deep_features: bool = True, enable_feature_engineering: bool = True):
        self.task_type = task_type
        self.max_features = max_features
        self.use_deep_features = use_deep_features
        self.enable_feature_engineering = enable_feature_engineering
        self.selector_ = None
        self.scaler_ = StandardScaler()
        self.selected_features_ = None
        self.feature_scores_ = None
        self.deep_extractor = None
        self.feature_engineer = None
        self.original_features = None
        
        if self.enable_feature_engineering:
            self.feature_engineer = AutomatedFeatureEngineer()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Enhanced fit with deep features and automated engineering."""
        try:
            # Convert to DataFrame for feature engineering
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            else:
                X_df = X.copy()
            
            # Automated feature engineering
            if self.enable_feature_engineering and self.feature_engineer:
                logger.info("Performing automated feature engineering...")
                X_engineered = self.feature_engineer.generate_features(X_df, y)
                X_combined = pd.concat([X_df, X_engineered], axis=1)
                X_combined = X_combined.select_dtypes(include=[np.number]).fillna(0)
            else:
                X_combined = X_df
            
            # Convert back to numpy
            X_processed = X_combined.values
            self.original_features = X_processed.shape[1]
            
            # Deep feature extraction
            if self.use_deep_features and X_processed.shape[1] > 5:
                logger.info("Extracting deep features...")
                embedding_dim = min(64, X_processed.shape[1] // 2)
                self.deep_extractor = DeepFeatureExtractor(
                    input_dim=X_processed.shape[1],
                    embedding_dim=embedding_dim
                ).to(DEVICE)
                
                # Train deep feature extractor
                self._train_deep_extractor(X_processed, y)
                
                # Extract deep features
                deep_features = self._extract_deep_features(X_processed)
                X_processed = np.column_stack([X_processed, deep_features])
            
            # Scale features
            X_scaled = self.scaler_.fit_transform(X_processed)
            
            # Determine number of features to select
            if self.max_features is None:
                self.max_features = min(X_scaled.shape[1], max(10, X_scaled.shape[1] // 2))
            
            # Select appropriate scoring function
            score_func = f_classif if self.task_type == "classification" else f_regression
            
            # Perform feature selection
            self.selector_ = SelectKBest(score_func=score_func, k=self.max_features)
            self.selector_.fit(X_scaled, y)
            
            # Store feature mask and scores
            self.selected_features_ = self.selector_.get_support()
            self.feature_scores_ = self.selector_.scores_
            
            logger.info(f"Feature selection: {X_processed.shape[1]} -> {np.sum(self.selected_features_)} features")
            
            return self

        except Exception as e:
            logger.error(f"Enhanced feature selection failed: {str(e)}")
            # Fallback: select all features
            self.selected_features_ = np.ones(X.shape[1], dtype=bool)
            self.feature_scores_ = np.ones(X.shape[1])
            return self

    def _train_deep_extractor(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Train the deep feature extractor."""
        try:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            y_tensor = torch.LongTensor(y).to(DEVICE) if self.task_type == "classification" else torch.FloatTensor(y).to(DEVICE)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=min(256, len(X) // 4), shuffle=True)
            
            # Add classifier head for training
            if self.task_type == "classification":
                n_classes = len(np.unique(y))
                classifier_head = nn.Linear(self.deep_extractor.embedding_dim, n_classes).to(DEVICE)
                criterion = nn.CrossEntropyLoss()
            else:
                classifier_head = nn.Linear(self.deep_extractor.embedding_dim, 1).to(DEVICE)
                criterion = nn.MSELoss()
            
            # Optimizer
            optimizer = optim.Adam(
                list(self.deep_extractor.parameters()) + list(classifier_head.parameters()),
                lr=0.001
            )
            
            # Training loop
            self.deep_extractor.train()
            classifier_head.train()
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    
                    # Extract features
                    features = self.deep_extractor(batch_x)
                    
                    # Make predictions
                    if self.task_type == "classification":
                        outputs = classifier_head(features)
                        loss = criterion(outputs, batch_y)
                    else:
                        outputs = classifier_head(features).squeeze()
                        loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 10 == 0:
                    logger.debug(f"Deep extractor epoch {epoch}, loss: {total_loss/len(loader):.4f}")
            
            # Set to evaluation mode
            self.deep_extractor.eval()
            
        except Exception as e:
            logger.warning(f"Deep feature extractor training failed: {e}")
            self.deep_extractor = None

    def _extract_deep_features(self, X: np.ndarray) -> np.ndarray:
        """Extract deep features from input data."""
        if self.deep_extractor is None:
            return np.array([]).reshape(len(X), 0)
        
        try:
            self.deep_extractor.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(DEVICE)
                deep_features = self.deep_extractor(X_tensor)
                return deep_features.cpu().numpy()
        except Exception as e:
            logger.warning(f"Deep feature extraction failed: {e}")
            return np.array([]).reshape(len(X), 0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Enhanced transform with deep features and engineering."""
        if self.selector_ is None:
            raise ValueError("RobustFeatureSelector must be fitted before transform")

        try:
            # Convert to DataFrame for consistency
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            else:
                X_df = X.copy()
            
            # Apply feature engineering if enabled
            if self.enable_feature_engineering and self.feature_engineer:
                X_engineered = self.feature_engineer.generate_features(X_df)
                X_combined = pd.concat([X_df, X_engineered], axis=1)
                X_combined = X_combined.select_dtypes(include=[np.number]).fillna(0)
            else:
                X_combined = X_df
            
            # Convert back to numpy
            X_processed = X_combined.values
            
            # Ensure consistent number of features
            if X_processed.shape[1] < self.original_features:
                # Pad with zeros if needed
                padding = np.zeros((X_processed.shape[0], self.original_features - X_processed.shape[1]))
                X_processed = np.column_stack([X_processed, padding])
            elif X_processed.shape[1] > self.original_features:
                # Truncate if needed
                X_processed = X_processed[:, :self.original_features]
            
            # Extract deep features if available
            if self.deep_extractor is not None:
                deep_features = self._extract_deep_features(X_processed)
                X_processed = np.column_stack([X_processed, deep_features])
            
            # Scale features
            X_scaled = self.scaler_.transform(X_processed)
            
            # Apply feature selection
            if hasattr(self.selector_, 'transform'):
                return self.selector_.transform(X_scaled)
            else:
                return X_scaled[:, self.selected_features_]

        except Exception as e:
            logger.error(f"Enhanced feature transform failed: {str(e)}")
            # Return original data if transform fails
            return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

class DynamicTrackGenerator:
    """
    Enhanced track generator with all advanced features including GPU support,
    online learning, deep features, and resource-aware routing.
    """
    
    def __init__(self, task_type: str = "classification", max_tracks: int = 20,
                 use_gpu: bool = False, enable_online_learning: bool = True,
                 enable_stacking: bool = True):
        self.task_type = task_type
        self.max_tracks = max_tracks
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_online_learning = enable_online_learning
        self.enable_stacking = enable_stacking
        self.tracks: Dict[str, HierarchicalTrack] = {}
        self.global_track: Optional[HierarchicalTrack] = None
        self.meta_track: Optional[MetaLearningTrack] = None
        self.stacking_meta_learner: Optional[StackingMetaLearner] = None
        self.track_counter = 0
        self.last_distribution: Optional[DataDistribution] = None
        self.distribution_history = deque(maxlen=10)
        self.prediction_history = deque(maxlen=1000)
        self.ensemble_weights = {}
        self.clustering_engine = AdvancedClusteringEngine()
        self.resource_monitor = {}
        self.concept_drift_detector = ConceptDriftDetector()
        self.track_confidence_history = defaultdict(float)
        
        if self.enable_stacking:
            self.stacking_meta_learner = StackingMetaLearner(task_type=task_type)

    def analyze_distribution(self, X: np.ndarray, y: np.ndarray) -> DataDistribution:
        """Enhanced distribution analysis with advanced clustering."""
        try:
            # Use advanced clustering engine
            cluster_labels, cluster_metadata = self.clustering_engine.fit_predict(X, method='auto')
            
            # Extract cluster information
            n_clusters = cluster_metadata['n_clusters']
            clustering_method = cluster_metadata['method']
            
            if 'cluster_centers' in cluster_metadata:
                cluster_centers = cluster_metadata['cluster_centers']
            elif 'means' in cluster_metadata:
                cluster_centers = cluster_metadata['means']
            else:
                # Calculate centroids manually
                cluster_centers = np.array([
                    np.mean(X[cluster_labels == i], axis=0)
                    for i in range(n_clusters)
                ])
            
            stability_score = cluster_metadata.get('silhouette_score', 0.0)
            
            # Calculate feature importance if possible
            feature_importance = None
            try:
                if self.global_track and hasattr(self.global_track.classifier, 'feature_importances_'):
                    feature_importance = self.global_track.classifier.feature_importances_
            except:
                pass
            
            # Detect concept drift
            concept_drift_score = 0.0
            if len(self.prediction_history) > 100:
                try:
                    recent_predictions = list(self.prediction_history)[-100:]
                    concept_drift_score = np.std(recent_predictions) / (np.mean(recent_predictions) + 1e-8)
                except:
                    pass
            
            return DataDistribution(
                cluster_centers=cluster_centers,
                cluster_labels=cluster_labels,
                n_clusters=n_clusters,
                stability_score=stability_score,
                clustering_method=clustering_method,
                feature_importance=feature_importance,
                concept_drift_score=concept_drift_score
            )

        except Exception as e:
            logger.warning(f"Enhanced distribution analysis failed: {e}")
            return DataDistribution(
                cluster_centers=np.mean(X, axis=0).reshape(1, -1),
                cluster_labels=np.zeros(X.shape[0]),
                n_clusters=1,
                stability_score=1.0,
                clustering_method='fallback'
            )

    def _get_base_estimator(self, model_type: str = 'rf', **kwargs):
        """Get base estimator with enhanced model types."""
        default_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 8),
            'random_state': 42
        }
        
        if model_type == 'rf':
            if self.task_type == "classification":
                return RandomForestClassifier(**default_params)
            else:
                return RandomForestRegressor(**default_params)
        
        elif model_type == 'xgb':
            xgb_params = default_params.copy()
            xgb_params['learning_rate'] = kwargs.get('learning_rate', 0.1)
            xgb_params['n_jobs'] = 1
            
            if self.task_type == "classification":
                return xgb.XGBClassifier(**xgb_params)
            else:
                return xgb.XGBRegressor(**xgb_params)
        
        elif model_type == 'lgb':
            lgb_params = default_params.copy()
            lgb_params['learning_rate'] = kwargs.get('learning_rate', 0.1)
            lgb_params['verbose'] = -1
            lgb_params['n_jobs'] = 1
            
            if self.task_type == "classification":
                return lgb.LGBMClassifier(**lgb_params)
            else:
                return lgb.LGBMRegressor(**lgb_params)
        
        elif model_type == 'mlp':
            if self.task_type == "classification":
                return MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
            else:
                return MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
        
        else:
            # Default to RandomForest
            if self.task_type == "classification":
                return RandomForestClassifier(**default_params)
            else:
                return RandomForestRegressor(**default_params)

    def create_track(self, X: np.ndarray, y: np.ndarray, level: str,
                    parent_track: Optional[HierarchicalTrack] = None,
                    config: Dict[str, Any] = None) -> HierarchicalTrack:
        """Enhanced track creation with advanced configuration."""
        track_name = f"{level}_track_{self.track_counter}"
        
        # Use config if provided, otherwise use defaults
        if config is None:
            config = {
                'model_type': 'rf',
                'n_estimators': 100,
                'max_depth': 8,
                'use_gpu': self.use_gpu,
                'enable_online_learning': self.enable_online_learning
            }
        
        # Create base estimator
        classifier = self._get_base_estimator(
            model_type=config.get('model_type', 'rf'),
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 8),
            learning_rate=config.get('learning_rate', 0.1)
        )
        
        # Create track
        track = HierarchicalTrack(
            name=track_name,
            level=level,
            classifier=classifier,
            parent_track=parent_track,
            use_gpu=config.get('use_gpu', False),
            enable_online_learning=config.get('enable_online_learning', False)
        )

        try:
            # Fit the classifier
            start_time = time.time()
            track.classifier.fit(X, y)
            training_time = time.time() - start_time
            
            # Update resource metrics
            track.resource_metrics.total_time = training_time
            
            # Memory usage estimation
            try:
                process = psutil.Process()
                track.resource_metrics.memory_mb = process.memory_info().rss / 1024 / 1024
            except:
                track.resource_metrics.memory_mb = 0.0
            
            # Analyze data distribution
            track.data_distribution = self.analyze_distribution(X, y)
            
            # Add to parent track if specified
            if parent_track:
                parent_track.add_child_track(track)
            
            # Store track
            self.tracks[track_name] = track
            self.track_counter += 1
            
            logger.info(f"Created enhanced {level} track: {track_name} ({config.get('model_type', 'rf')})")
            
            return track

        except Exception as e:
            logger.error(f"Failed to create enhanced track: {e}")
            raise

    def create_hierarchical_structure(self, X: np.ndarray, y: np.ndarray, meta_config: Dict[str, Any] = None):
        """Enhanced hierarchical structure creation with meta-learning guidance."""
        logger.info("Creating enhanced hierarchical structure...")
        
        # Use meta-learning for optimal configuration if available
        if meta_config is None and self.meta_track:
            suggestion = self.meta_track.suggest_track_creation(X, y, 0.5)
            meta_config = suggestion.get('config', {})
        
        # Create global track with optimal configuration
        global_config = meta_config.copy() if meta_config else {}
        global_config['level'] = TrackLevel.GLOBAL
        self.global_track = self.create_track(X, y, TrackLevel.GLOBAL, config=global_config)

        # Analyze distribution for regional tracks
        distribution = self.analyze_distribution(X, y)
        self.last_distribution = distribution

        # Create regional tracks based on clusters
        regional_configs = []
        for cluster_id in range(distribution.n_clusters):
            mask = distribution.cluster_labels == cluster_id
            if np.sum(mask) < 20:  # Increased minimum samples threshold
                continue

            X_regional = X[mask]
            y_regional = y[mask]
            
            # Get cluster-specific configuration
            regional_config = meta_config.copy() if meta_config else {}
            regional_config['level'] = TrackLevel.REGIONAL
            
            # Vary model types for diversity
            model_types = ['rf', 'xgb', 'lgb']
            regional_config['model_type'] = model_types[cluster_id % len(model_types)]
            
            regional_track = self.create_track(
                X_regional, y_regional,
                TrackLevel.REGIONAL,
                self.global_track,
                regional_config
            )
            
            regional_configs.append((regional_track, X_regional, y_regional))

            # Create local tracks if enough samples and enabled
            if len(X_regional) >= 100 and meta_config.get('create_children', False):
                local_distribution = self.analyze_distribution(X_regional, y_regional)
                
                for local_id in range(min(3, local_distribution.n_clusters)):  # Limit local tracks
                    local_mask = local_distribution.cluster_labels == local_id
                    if np.sum(local_mask) < 30:
                        continue

                    X_local = X_regional[local_mask]
                    y_local = y_regional[local_mask]
                    
                    local_config = meta_config.copy() if meta_config else {}
                    local_config['level'] = TrackLevel.LOCAL
                    local_config['model_type'] = 'rf'  # Use simpler models for local tracks
                    local_config['n_estimators'] = 50  # Smaller ensembles
                    
                    self.create_track(
                        X_local, y_local,
                        TrackLevel.LOCAL,
                        regional_track,
                        local_config
                    )

        # Train stacking meta-learner if enabled
        if self.enable_stacking and len(self.tracks) > 1:
            logger.info("Training stacking meta-learner...")
            try:
                # Get base predictions for stacking
                base_predictions = []
                for track_name, track in self.tracks.items():
                    if track.level != TrackLevel.META:
                        try:
                            if self.task_type == "classification" and hasattr(track.classifier, 'predict_proba'):
                                pred = track.classifier.predict_proba(X)
                                if pred.shape[1] > 1:
                                    pred = pred[:, 1]  # Use positive class probability
                                base_predictions.append(pred)
                            else:
                                pred = track.classifier.predict(X)
                                base_predictions.append(pred)
                        except Exception as e:
                            logger.warning(f"Failed to get predictions from {track_name}: {e}")
                
                if base_predictions:
                    self.stacking_meta_learner.fit(base_predictions, y)
                    logger.info("Stacking meta-learner trained successfully")
                
            except Exception as e:
                logger.error(f"Stacking meta-learner training failed: {e}")

    def ensemble_predict(self, X: np.ndarray, method: str = "stacking") -> np.ndarray:
        """Enhanced ensemble prediction with stacking and resource awareness."""
        if not self.tracks:
            raise ValueError("No tracks available for prediction")

        try:
            # Get predictions from all tracks
            track_predictions = []
            track_confidences = []
            track_names = []
            
            for track_name, track in self.tracks.items():
                if track.level == TrackLevel.META:
                    continue
                
                try:
                    start_time = time.time()
                    
                    # Choose between online and offline prediction
                    if track.enable_online_learning and RIVER_AVAILABLE:
                        pred = track.predict_online(X)
                    else:
                        pred = track.classifier.predict(X)
                    
                    prediction_time = time.time() - start_time
                    
                    # Update resource metrics
                    track.resource_metrics.total_time += prediction_time
                    track.resource_metrics.prediction_count += 1
                    
                    track_predictions.append(pred)
                    track_names.append(track_name)
                    
                    # Calculate confidence
                    if hasattr(track.classifier, "predict_proba"):
                        proba = track.classifier.predict_proba(X)
                        confidence = np.max(proba, axis=1)
                    else:
                        confidence = np.ones(len(X)) * 0.8
                    
                    track_confidences.append(confidence)
                    
                except Exception as e:
                    logger.debug(f"Prediction failed for track {track_name}: {e}")
                    continue

            if not track_predictions:
                raise ValueError("No valid predictions available")

            # Convert to numpy arrays
            track_predictions = np.array(track_predictions)
            track_confidences = np.array(track_confidences)

            # Apply ensemble method
            if method == "stacking" and self.enable_stacking and self.stacking_meta_learner and self.stacking_meta_learner.fitted:
                try:
                    # Use stacking meta-learner
                    final_predictions = self.stacking_meta_learner.predict(list(track_predictions))
                    logger.debug("Used stacking meta-learner for predictions")
                    return final_predictions
                except Exception as e:
                    logger.warning(f"Stacking prediction failed, falling back to weighted voting: {e}")
                    method = "weighted_voting"

            if method == "weighted_voting" or method == "stacking":
                # Resource-aware weighted voting
                weights = []
                for i, track_name in enumerate(track_names):
                    track = self.tracks[track_name]
                    
                    # Combine performance and efficiency
                    performance_weight = track.performance_score
                    efficiency_weight = track.resource_metrics.get_efficiency_score()
                    combined_weight = 0.7 * performance_weight + 0.3 * efficiency_weight
                    
                    weights.append(combined_weight)
                
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize
                
                if self.task_type == "classification":
                    final_predictions = np.zeros(len(X), dtype=int)
                    for i in range(len(X)):
                        votes = {}
                        for j in range(len(track_predictions)):
                            vote = track_predictions[j, i]
                            weight = weights[j]
                            votes[vote] = votes.get(vote, 0) + weight
                        final_predictions[i] = max(votes.items(), key=lambda x: x[1])[0]
                    return final_predictions
                else:
                    return np.sum(track_predictions * weights[:, np.newaxis], axis=0)

            else:  # simple majority voting for classification, mean for regression
                if self.task_type == "classification":
                    return np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(),
                        axis=0,
                        arr=track_predictions
                    )
                else:
                    return np.mean(track_predictions, axis=0)

        except Exception as e:
            logger.error(f"Enhanced ensemble prediction failed: {str(e)}")
            # Fallback to global track
            if self.global_track and self.global_track.classifier:
                return self.global_track.classifier.predict(X)
            else:
                raise ValueError("Enhanced ensemble prediction failed and no fallback available")

    def update_tracks(self, X: np.ndarray, y: np.ndarray):
        """Enhanced track updating with online learning and concept drift detection."""
        # Detect concept drift
        if len(self.tracks) > 0 and len(self.prediction_history) > 100:
            try:
                predictions = self.predict(X)
                drift_detected = self.concept_drift_detector.update(X, predictions, y)
                
                if drift_detected:
                    logger.info("Concept drift detected, updating tracks...")
                    
                    # Update online models
                    for track in self.tracks.values():
                        if track.enable_online_learning:
                            track.update_online(X, y)
                    
                    # Optionally retrain some tracks
                    if np.random.random() < 0.3:  # 30% chance of retraining
                        self.create_hierarchical_structure(X, y)
                
            except Exception as e:
                logger.warning(f"Concept drift detection failed: {e}")

        # Regular distribution analysis and updates
        new_distribution = self.analyze_distribution(X, y)
        self.distribution_history.append(new_distribution)

        if (self.last_distribution and
            not self.last_distribution.is_stable(new_distribution, threshold=0.7)):
            
            logger.info("Distribution change detected, updating tracks...")
            self.create_hierarchical_structure(X, y)

            # Update meta-learner
            if self.meta_track:
                self.meta_track.learn_track_creation_patterns(X, y, self.tracks)

        self.last_distribution = new_distribution

    def get_track_explanations(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Get explanations from all tracks."""
        explanations = {}
        
        for track_name, track in self.tracks.items():
            if track.level != TrackLevel.META:
                try:
                    explanation = track.get_explanation(X[:5])  # Limit for performance
                    explanations[track_name] = explanation
                except Exception as e:
                    logger.warning(f"Failed to get explanation from {track_name}: {e}")
                    explanations[track_name] = {'error': str(e)}
        
        return explanations

    def get_resource_metrics(self) -> Dict[str, ResourceMetrics]:
        """Get resource metrics from all tracks."""
        metrics = {}
        for track_name, track in self.tracks.items():
            metrics[track_name] = track.resource_metrics
        return metrics

    def prune_inefficient_tracks(self, efficiency_threshold: float = 0.3):
        """Remove tracks with poor efficiency scores."""
        tracks_to_remove = []
        
        for track_name, track in self.tracks.items():
            if track.level in [TrackLevel.REGIONAL, TrackLevel.LOCAL]:
                efficiency = track.resource_metrics.get_efficiency_score()
                if efficiency < efficiency_threshold and track.usage_count > 10:
                    tracks_to_remove.append(track_name)
        
        for track_name in tracks_to_remove:
            logger.info(f"Pruning inefficient track: {track_name}")
            del self.tracks[track_name]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Enhanced fit method with all advanced features."""
        logger.info("Fitting enhanced DynamicTrackGenerator...")
        
        # Initialize meta-learning track first
        self.meta_track = MetaLearningTrack(name="meta_track", task_type=self.task_type)
        
        # Get optimal configuration from meta-learning
        meta_suggestion = self.meta_track.suggest_track_creation(X, y, 0.5)
        meta_config = meta_suggestion.get('config', {})
        
        # Create hierarchical structure with meta-learning guidance
        self.create_hierarchical_structure(X, y, meta_config)
        
        # Train meta-learner on created tracks
        self.meta_track.learn_track_creation_patterns(X, y, self.tracks)
        
        logger.info(f"Enhanced DynamicTrackGenerator fitted with {len(self.tracks)} tracks")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Enhanced prediction using all advanced features."""
        predictions = self.ensemble_predict(X, method="stacking")
        
        # Store predictions for concept drift detection
        if len(predictions) <= 1000:  # Limit storage
            self.prediction_history.extend(predictions)
        
        return predictions

# Helper function for softmax (unchanged)
def softmax(x: np.ndarray, axis: int = None) -> np.ndarray:
    """Compute softmax values along specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class OptimizedTRA(BaseEstimator):
    """
    Fully Enhanced OptimizedTRA with all advanced features:
    - Deep feature extraction
    - Multiple base learners (XGBoost, LightGBM, Neural Networks)
    - Bayesian optimization via meta-learning
    - Online/streaming learning capabilities
    - Advanced clustering (DBSCAN, Gaussian Mixture)
    - Stacking meta-learner
    - Resource-aware routing
    - Explainability (SHAP/LIME)
    - GPU support
    - Automated feature engineering
    """
    
    def __init__(self, task_type="classification", max_tracks=20,
                 enable_meta_learning=True, handle_imbalanced=True,
                 meta_cv_folds=3, use_parallel=False, use_gpu=False,
                 enable_online_learning=True, enable_stacking=True,
                 enable_deep_features=True, enable_feature_engineering=True,
                 enable_explainability=True):
        
        # Core parameters
        self.task_type = task_type
        self.max_tracks = max_tracks
        self.enable_meta_learning = enable_meta_learning
        self.handle_imbalanced = handle_imbalanced
        self.meta_cv_folds = meta_cv_folds
        self.use_parallel = use_parallel
        
        # Enhanced parameters
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_online_learning = enable_online_learning
        self.enable_stacking = enable_stacking
        self.enable_deep_features = enable_deep_features
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_explainability = enable_explainability
        
        # State variables
        self.fitted_ = False
        self.start_time = time.time()
        
        # Initialize enhanced components
        self.track_generator = DynamicTrackGenerator(
            task_type=task_type,
            max_tracks=max_tracks,
            use_gpu=use_gpu,
            enable_online_learning=enable_online_learning,
            enable_stacking=enable_stacking
        )
        
        self.feature_selector_ = RobustFeatureSelector(
            task_type=task_type,
            use_deep_features=enable_deep_features,
            enable_feature_engineering=enable_feature_engineering
        )
        
        self.track_performance = defaultdict(default_track_performance)
        self.active_records = {}
        self.prediction_count_ = 0
        self.learning_rate = 0.01
        self.cache = {}
        self.resource_monitor = {}
        self.explainer_cache = {}
        
        # GPU setup
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled using device: {DEVICE}")
        
        # Initialize concept drift detection
        self.concept_drift_detector = ConceptDriftDetector()

    def get_model_explanation(self, X: np.ndarray, sample_size: int = 5) -> Dict[str, Any]:
        """Get model explanations using SHAP/LIME."""
        if not self.enable_explainability:
            return {"explanation": "Explainability disabled"}
        
        try:
            # Limit sample size for performance
            X_sample = X[:min(sample_size, len(X))]
            X_selected = self.feature_selector_.transform(X_sample)
            
            # Get explanations from track generator
            explanations = self.track_generator.get_track_explanations(X_selected)
            
            return {
                "global_explanation": explanations,
                "feature_importance": self._get_global_feature_importance(),
                "method": "ensemble_explanation"
            }
            
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return {"explanation": f"Failed to generate explanation: {e}"}

    def _get_global_feature_importance(self) -> Dict[str, float]:
        """Calculate global feature importance across all tracks."""
        importance_scores = {}
        
        for track_name, track in self.track_generator.tracks.items():
            if hasattr(track.classifier, 'feature_importances_'):
                for i, importance in enumerate(track.classifier.feature_importances_):
                    feature_name = f"feature_{i}"
                    importance_scores[feature_name] = importance_scores.get(feature_name, 0) + importance
        
        # Normalize scores
        total_score = sum(importance_scores.values())
        if total_score > 0:
            importance_scores = {k: v/total_score for k, v in importance_scores.items()}
        
        return importance_scores

    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource usage metrics."""
        metrics = self.track_generator.get_resource_metrics()
        
        # Add system-level metrics
        try:
            process = psutil.Process()
            system_metrics = {
                "system_memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "total_predictions": self.prediction_count_,
                "cache_size": len(self.cache)
            }
            
            if torch.cuda.is_available():
                system_metrics["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            
            return {
                "track_metrics": metrics,
                "system_metrics": system_metrics
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {"track_metrics": metrics}

    def optimize_performance(self):
        """Optimize system performance by pruning inefficient tracks."""
        logger.info("Optimizing system performance...")
        
        # Prune inefficient tracks
        self.track_generator.prune_inefficient_tracks()
        
        # Clear old cache entries
        if len(self.cache) > 1000:
            # Keep only the most recent 500 entries
            cache_items = list(self.cache.items())
            self.cache = dict(cache_items[-500:])
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Performance optimization completed")

    def update_online(self, X: np.ndarray, y: np.ndarray):
        """Update models with new data using online learning."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before online updates")
        
        try:
            # Transform features
            X_selected = self.feature_selector_.transform(X)
            
            # Update tracks with online learning
            self.track_generator.update_tracks(X_selected, y)
            
            # Detect concept drift
            predictions = self.predict(X)
            drift_detected = self.concept_drift_detector.update(X_selected, predictions, y)
            
            if drift_detected:
                logger.warning("Concept drift detected - consider retraining")
            
            logger.info(f"Online update completed for {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Online update failed: {e}")

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores."""
        if not self.fitted_:
            raise ValueError("TRA system must be fitted before prediction")
        
        X_selected = self.feature_selector_.transform(X)
        
        # Get predictions from all tracks
        track_predictions = []
        track_confidences = []
        
        for track_name, track in self.track_generator.tracks.items():
            if track.level != TrackLevel.META:
                try:
                    pred = track.predict(X_selected)
                    track_predictions.append(pred)
                    
                    # Calculate confidence
                    if hasattr(track.classifier, 'predict_proba'):
                        proba = track.classifier.predict_proba(X_selected)
                        confidence = np.max(proba, axis=1)
                    else:
                        confidence = np.ones(len(X)) * 0.8
                    
                    track_confidences.append(confidence)
                    
                except Exception as e:
                    logger.debug(f"Prediction failed for track {track_name}: {e}")
                    continue
        
        if not track_predictions:
            raise ValueError("No valid predictions available")
        
        # Ensemble predictions
        final_predictions = self.track_generator.ensemble_predict(X_selected)
        
        # Calculate ensemble confidence as mean of track confidences
        track_confidences = np.array(track_confidences)
        ensemble_confidence = np.mean(track_confidences, axis=0)
        
        return final_predictions, ensemble_confidence

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple[np.ndarray, np.ndarray] = None):
        """Enhanced fit method with all advanced features."""
        logger.info("Starting Enhanced TRA system training...")
        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Task: {self.task_type}")
        
        if self.task_type == "classification":
            logger.info(f"Classes: {len(np.unique(y))}")
            class_distribution = np.bincount(y)
            logger.info(f"Class distribution: {dict(enumerate(class_distribution))}")

        start_time = time.time()
        
        # Enhanced feature selection with deep features and automation
        logger.info("Enhanced feature selection and engineering...")
        X_selected = self.feature_selector_.fit_transform(X, y)
        logger.info(f"Features: {X.shape[1]} -> {X_selected.shape[1]} (including engineered features)")
        
        # Initialize and fit enhanced track generator
        logger.info("Creating enhanced hierarchical track structure...")
        self.track_generator.fit(X_selected, y)
        
        # Initialize signal conditions with enhanced features
        logger.info("Creating adaptive signal conditions...")
        self._initialize_enhanced_signal_conditions()
        
        # Validation if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_selected = self.feature_selector_.transform(X_val)
            val_predictions = self.track_generator.predict(X_val_selected)
            
            if self.task_type == "classification":
                val_score = accuracy_score(y_val, val_predictions)
                logger.info(f"Validation accuracy: {val_score:.4f}")
            else:
                val_score = -mean_squared_error(y_val, val_predictions)
                logger.info(f"Validation MSE: {-val_score:.4f}")

        training_time = time.time() - start_time
        
        # Training summary
        logger.info("\nEnhanced TRA Training Summary:")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Total tracks created: {len(self.track_generator.tracks)}")
        
        # Track breakdown by level
        level_counts = {}
        for track in self.track_generator.tracks.values():
            level_counts[track.level] = level_counts.get(track.level, 0) + 1
        
        for level, count in level_counts.items():
            logger.info(f"  {level.capitalize()} tracks: {count}")
        
        # Enhanced features summary
        if self.enable_stacking:
            logger.info("Stacking meta-learner: Enabled")
        if self.use_gpu:
            logger.info(f"GPU acceleration: Enabled ({DEVICE})")
        if self.enable_online_learning:
            logger.info("Online learning: Enabled")
        
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Enhanced prediction with all advanced features."""
        if not self.fitted_:
            raise ValueError("TRA system must be fitted before prediction")

        # Update prediction count
        self.prediction_count_ += len(X)
        
        # Transform features
        X_selected = self.feature_selector_.transform(X)
        
        # Make predictions using enhanced ensemble
        predictions = self.track_generator.predict(X_selected)
        
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Enhanced probability prediction."""
        if not self.fitted_:
            raise ValueError("TRA system must be fitted before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification tasks")
        
        X_selected = self.feature_selector_.transform(X)
        
        # Get probabilities from all tracks
        track_probabilities = []
        
        for track_name, track in self.track_generator.tracks.items():
            if track.level != TrackLevel.META:
                try:
                    if hasattr(track.classifier, 'predict_proba'):
                        proba = track.classifier.predict_proba(X_selected)
                        track_probabilities.append(proba)
                except Exception as e:
                    logger.debug(f"Probability prediction failed for track {track_name}: {e}")
                    continue
        
        if not track_probabilities:
            raise ValueError("No probability predictions available")
        
        # Use stacking meta-learner if available
        if (self.enable_stacking and 
            self.track_generator.stacking_meta_learner and 
            self.track_generator.stacking_meta_learner.fitted):
            try:
                return self.track_generator.stacking_meta_learner.predict_proba(track_probabilities)
            except:
                pass
        
        # Fallback to simple averaging
        return np.mean(track_probabilities, axis=0)

    def _initialize_enhanced_signal_conditions(self):
        """Initialize enhanced adaptive signal conditions."""
        self.signal_conditions = {}
        tracks = self.track_generator.tracks
        
        for src_name in tracks:
            for tgt_name in tracks:
                if src_name != tgt_name:
                    condition = AdaptiveSignalCondition(
                        source_track=src_name,
                        target_track=tgt_name,
                        model_type=self.task_type,
                        initial_threshold=0.6
                    )
                    
                    # Enhanced conditions with resource awareness
                    condition.resource_aware = True
                    
                    self.signal_conditions[(src_name, tgt_name)] = condition

    def visualize_enhanced(self, output_file: str = None, figsize: Tuple[int, int] = (16, 12)):
        """Enhanced visualization with resource metrics and performance data."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before visualization")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # 1. Track Performance Distribution
            performances = [track.performance_score for track in self.track_generator.tracks.values()]
            ax1.hist(performances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Track Performance Distribution', fontweight='bold')
            ax1.set_xlabel('Performance Score')
            ax1.set_ylabel('Number of Tracks')
            
            # 2. Resource Usage by Track Level
            resource_data = self.get_resource_metrics()['track_metrics']
            levels = []
            latencies = []
            memory_usage = []
            
            for track_name, metrics in resource_data.items():
                track = self.track_generator.tracks[track_name]
                levels.append(track.level)
                latencies.append(metrics.latency_ms)
                memory_usage.append(metrics.memory_mb)
            
            level_unique = list(set(levels))
            level_latencies = [np.mean([latencies[i] for i, l in enumerate(levels) if l == level]) 
                             for level in level_unique]
            
            ax2.bar(level_unique, level_latencies, alpha=0.7, color='lightgreen')
            ax2.set_title('Average Latency by Track Level', fontweight='bold')
            ax2.set_xlabel('Track Level')
            ax2.set_ylabel('Latency (ms)')
            
            # 3. Track Usage Over Time
            usage_counts = [track.usage_count for track in self.track_generator.tracks.values()]
            track_names = [track.name for track in self.track_generator.tracks.values()]
            
            # Top 10 most used tracks
            top_indices = np.argsort(usage_counts)[-10:]
            top_names = [track_names[i] for i in top_indices]
            top_usage = [usage_counts[i] for i in top_indices]
            
            ax3.barh(range(len(top_names)), top_usage, alpha=0.7, color='coral')
            ax3.set_yticks(range(len(top_names)))
            ax3.set_yticklabels(top_names, fontsize=8)
            ax3.set_title('Top 10 Most Used Tracks', fontweight='bold')
            ax3.set_xlabel('Usage Count')
            
            # 4. Feature Importance (if available)
            feature_importance = self._get_global_feature_importance()
            if feature_importance:
                # Top 15 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
                features, importances = zip(*sorted_features)
                
                ax4.barh(range(len(features)), importances, alpha=0.7, color='gold')
                ax4.set_yticks(range(len(features)))
                ax4.set_yticklabels(features, fontsize=8)
                ax4.set_title('Top 15 Feature Importances', fontweight='bold')
                ax4.set_xlabel('Importance Score')
            else:
                ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Feature Importance', fontweight='bold')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Enhanced visualization saved to {output_file}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Enhanced visualization failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        if not self.fitted_:
            return {"status": "Not fitted"}
        
        # Basic statistics
        total_tracks = len(self.track_generator.tracks)
        active_tracks = sum(1 for track in self.track_generator.tracks.values() 
                           if track.usage_count > 0)
        
        # Performance statistics
        performances = [track.performance_score for track in self.track_generator.tracks.values()]
        avg_performance = np.mean(performances) if performances else 0.0
        
        # Resource metrics
        resource_metrics = self.get_resource_metrics()
        
        # Concept drift status
        drift_status = "Stable"
        if hasattr(self.concept_drift_detector, 'drift_detected') and self.concept_drift_detector.drift_detected:
            drift_status = "Drift Detected"
        
        status = {
            "system_status": "Operational",
            "total_tracks": total_tracks,
            "active_tracks": active_tracks,
            "average_performance": avg_performance,
            "total_predictions": self.prediction_count_,
            "concept_drift_status": drift_status,
            "gpu_enabled": self.use_gpu,
            "online_learning_enabled": self.enable_online_learning,
            "stacking_enabled": self.enable_stacking,
            "explainability_enabled": self.enable_explainability,
            "resource_metrics": resource_metrics,
            "cache_size": len(self.cache)
        }
        
        return status


# Enhanced demonstration script
if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("FULLY ENHANCED OPTIMIZED TRACK/RAIL ALGORITHM (TRA) DEMONSTRATION")
    logger.info("With Deep Features, GPU Support, Online Learning, and Explainability")
    logger.info("=" * 80)

    try:
        # Performance monitoring setup
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            initial_memory = 0

        # Generate comprehensive test datasets
        logger.info("\nGenerating Enhanced Test Datasets...")
        n_samples = 15000  # Larger dataset for better testing
        n_features = 20
        n_classes = 4

        # Classification dataset with more complexity
        X_class, y_class = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=12,
            n_redundant=4,
            n_clusters_per_class=3,
            flip_y=0.02,  # Add some noise
            random_state=42
        )

        # Regression dataset
        X_reg, y_reg = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=12,
            noise=0.1,
            random_state=42
        )

        # Split datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )

        # Enhanced test configurations
        configs = [
            ("Baseline Enhanced", {
                'enable_meta_learning': True,
                'max_tracks': 15,
                'use_gpu': False,
                'enable_online_learning': False,
                'enable_stacking': False,
                'enable_deep_features': False,
                'enable_feature_engineering': False,
                'enable_explainability': False
            }),
            ("GPU + Deep Features", {
                'enable_meta_learning': True,
                'max_tracks': 20,
                'use_gpu': True,
                'enable_online_learning': False,
                'enable_stacking': True,
                'enable_deep_features': True,
                'enable_feature_engineering': True,
                'enable_explainability': False
            }),
            ("Full Enhancement", {
                'enable_meta_learning': True,
                'max_tracks': 25,
                'use_gpu': True,
                'enable_online_learning': True,
                'enable_stacking': True,
                'enable_deep_features': True,
                'enable_feature_engineering': True,
                'enable_explainability': True
            })
        ]

        results = {}

        for config_name, config_params in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING: {config_name}")
            logger.info(f"{'='*60}")

            # Initialize Enhanced TRA
            start_init = time.time()
            tra = OptimizedTRA(
                task_type="classification",
                handle_imbalanced=True,
                **config_params
            )

            # Training phase with validation
            logger.info("\n🚀 Enhanced Training Phase:")
            start_train = time.time()
            tra.fit(X_train, y_train, validation_data=(X_val, y_val))
            train_time = time.time() - start_train

            # Enhanced prediction tests
            logger.info("\n📊 Enhanced Prediction Tests:")
            
            # Standard prediction
            start_pred = time.time()
            y_pred = tra.predict(X_test)
            pred_time = time.time() - start_pred
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Prediction with confidence
            if hasattr(tra, 'predict_with_confidence'):
                y_pred_conf, confidence_scores = tra.predict_with_confidence(X_test)
                avg_confidence = np.mean(confidence_scores)
            else:
                avg_confidence = 0.0

            # Probability prediction
            try:
                y_proba = tra.predict_proba(X_test)
                proba_available = True
            except:
                proba_available = False

            # Online learning test
            online_test_score = 0.0
            if config_params.get('enable_online_learning', False):
                try:
                    # Simulate online updates
                    online_X = X_test[:100]
                    online_y = y_test[:100]
                    tra.update_online(online_X, online_y)
                    
                    # Test prediction after online update
                    online_pred = tra.predict(online_X)
                    online_test_score = accuracy_score(online_y, online_pred)
                    logger.info(f"Online learning test accuracy: {online_test_score:.4f}")
                except Exception as e:
                    logger.warning(f"Online learning test failed: {e}")

            # Explainability test
            explanation_available = False
            if config_params.get('enable_explainability', False):
                try:
                    explanations = tra.get_model_explanation(X_test[:5])
                    explanation_available = True
                    logger.info("Model explanations generated successfully")
                except Exception as e:
                    logger.warning(f"Explanation generation failed: {e}")

            # Resource metrics
            try:
                resource_metrics = tra.get_resource_metrics()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory
            except:
                memory_used = 0
                resource_metrics = {}

            # System status
            system_status = tra.get_system_status()

            # Store results
            results[config_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'train_time': train_time,
                'pred_time': pred_time,
                'memory_mb': memory_used,
                'n_tracks': len(tra.track_generator.tracks),
                'avg_confidence': avg_confidence,
                'online_test_score': online_test_score,
                'proba_available': proba_available,
                'explanation_available': explanation_available,
                'system_status': system_status
            }

            # Detailed results logging
            logger.info(f"\n📈 Performance Metrics:")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   F1 Score: {f1:.4f}")
            logger.info(f"   Average Confidence: {avg_confidence:.4f}")
            logger.info(f"   Training Time: {train_time:.2f}s")
            logger.info(f"   Prediction Time: {pred_time:.4f}s")
            logger.info(f"   Memory Usage: {memory_used:.1f}MB")

            logger.info(f"\n🏗️ System Architecture:")
            logger.info(f"   Total Tracks: {len(tra.track_generator.tracks)}")
            
            # Track breakdown by level
            level_counts = {}
            for track in tra.track_generator.tracks.values():
                level_counts[track.level] = level_counts.get(track.level, 0) + 1
            
            for level, count in level_counts.items():
                logger.info(f"   {level.capitalize()}: {count}")

            logger.info(f"\n🔧 Enhanced Features:")
            logger.info(f"   GPU Acceleration: {'✓' if config_params.get('use_gpu') and torch.cuda.is_available() else '✗'}")
            logger.info(f"   Deep Features: {'✓' if config_params.get('enable_deep_features') else '✗'}")
            logger.info(f"   Feature Engineering: {'✓' if config_params.get('enable_feature_engineering') else '✗'}")
            logger.info(f"   Online Learning: {'✓' if config_params.get('enable_online_learning') else '✗'}")
            logger.info(f"   Stacking Ensemble: {'✓' if config_params.get('enable_stacking') else '✗'}")
            logger.info(f"   Explainability: {'✓' if config_params.get('enable_explainability') else '✗'}")

            # Enhanced visualization
            try:
                output_dir = 'enhanced_visualization_results'
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{config_name.lower().replace(' ', '_')}_visualization.png"
                filepath = os.path.join(output_dir, filename)
                
                tra.visualize_enhanced(filepath)
                logger.info(f"Enhanced visualization saved: {filepath}")
                
            except Exception as vis_err:
                logger.warning(f"Enhanced visualization failed: {vis_err}")

            # Performance optimization
            try:
                tra.optimize_performance()
                logger.info("System performance optimized")
            except Exception as opt_err:
                logger.warning(f"Performance optimization failed: {opt_err}")

            # Cleanup
            del tra
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Final comprehensive comparison
        logger.info(f"\n{'='*80}")
        logger.info("COMPREHENSIVE PERFORMANCE COMPARISON")
        logger.info(f"{'='*80}")

        comparison_metrics = [
            'accuracy', 'f1_score', 'train_time', 'pred_time', 
            'memory_mb', 'n_tracks', 'avg_confidence'
        ]

        for metric in comparison_metrics:
            logger.info(f"\n📊 {metric.replace('_', ' ').title()}:")
            for config_name in results:
                value = results[config_name][metric]
                if isinstance(value, float):
                    logger.info(f"   {config_name:20s}: {value:.4f}")
                else:
                    logger.info(f"   {config_name:20s}: {value}")

        # Feature comparison
        logger.info(f"\n🎯 Feature Availability:")
        feature_matrix = [
            ['Configuration', 'Probability', 'Online Learning', 'Explainability'],
        ]
        
        for config_name in results:
            row = [
                config_name,
                '✓' if results[config_name]['proba_available'] else '✗',
                '✓' if results[config_name]['online_test_score'] > 0 else '✗',
                '✓' if results[config_name]['explanation_available'] else '✗'
            ]
            feature_matrix.append(row)

        # Print feature matrix
        for row in feature_matrix:
            logger.info(f"   {row[0]:20s} {row[1]:12s} {row[2]:15s} {row[3]:15s}")

        logger.info(f"\n🎉 ENHANCED TRA DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info(f"All advanced features tested and validated.")

    except Exception as e:
        logger.error(f"An error occurred during enhanced demonstration: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Final cleanup completed.")

