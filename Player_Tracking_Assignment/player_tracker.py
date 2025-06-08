import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from collections import defaultdict, deque
import pickle
import os
from typing import Dict, List, Tuple, Optional
import time

class PlayerTracker:
    """
    Advanced Player Tracking and Re-identification System
    
    This class implements a comprehensive solution for tracking players in sports videos,
    maintaining consistent IDs even when players go out of frame and reappear.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the PlayerTracker
        
        Args:
            model_path: Path to the YOLOv11 model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Tracking parameters
        self.max_disappeared = 30  # Max frames a player can be missing
        self.max_distance = 100    # Max distance for matching detections
        self.feature_history_size = 10  # Number of features to store per player
        
        # Player tracking data structures
        self.players = {}  # Active players: {id: PlayerData}
        self.disappeared_players = {}  # Recently disappeared players
        self.next_player_id = 1
        self.frame_count = 0
        
        # Feature extraction for re-identification
        self.feature_extractor = FeatureExtractor()
        
        # Performance metrics
        self.processing_times = []
        self.detection_counts = []
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players and ball in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': cls,
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    })
        
        return detections
    
    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract features from player region for re-identification
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector for the player
        """
        x1, y1, x2, y2 = bbox
        player_crop = frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return np.zeros(self.feature_extractor.feature_dim)
        
        features = self.feature_extractor.extract(player_crop)
        return features
    
    def match_detections_to_players(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, Dict]:
        """
        Match current detections to existing players using position and appearance
        
        Args:
            detections: List of current frame detections
            frame: Current video frame
            
        Returns:
            Dictionary mapping player IDs to matched detections
        """
        if not detections:
            return {}
        
        # Filter only player detections (class 2 is player in this model)
        player_detections = [d for d in detections if d['class'] == 2]
        
        if not player_detections:
            return {}
        
        matched_players = {}
        unmatched_detections = list(range(len(player_detections)))
        
        # Get current player centers and features
        current_centers = np.array([d['center'] for d in player_detections])
        current_features = []
        
        for detection in player_detections:
            features = self.extract_features(frame, detection['bbox'])
            current_features.append(features)
        current_features = np.array(current_features)
        
        # Match with existing active players
        if self.players:
            player_ids = list(self.players.keys())
            player_centers = np.array([self.players[pid]['last_center'] for pid in player_ids])
            player_features = np.array([self.players[pid]['avg_features'] for pid in player_ids])
            
            # Calculate distance matrix (position + appearance)
            position_dist = cdist(current_centers, player_centers)
            appearance_dist = cdist(current_features, player_features, metric='cosine')
            
            # Combine distances with weights
            combined_dist = 0.6 * position_dist + 0.4 * appearance_dist
            
            # Hungarian algorithm for optimal matching
            matches = self._hungarian_matching(combined_dist, self.max_distance)
            
            for det_idx, player_idx in matches:
                if det_idx in unmatched_detections:
                    player_id = player_ids[player_idx]
                    matched_players[player_id] = player_detections[det_idx]
                    unmatched_detections.remove(det_idx)
        
        # Try to match with recently disappeared players
        if unmatched_detections and self.disappeared_players:
            for det_idx in unmatched_detections.copy():
                best_match_id = None
                best_match_score = float('inf')
                
                for player_id, player_data in self.disappeared_players.items():
                    # Calculate appearance similarity
                    appearance_dist = np.linalg.norm(
                        current_features[det_idx] - player_data['avg_features']
                    )
                    
                    if appearance_dist < best_match_score and appearance_dist < 0.7:
                        best_match_score = appearance_dist
                        best_match_id = player_id
                
                if best_match_id is not None:
                    # Reactivate the player
                    matched_players[best_match_id] = player_detections[det_idx]
                    self.players[best_match_id] = self.disappeared_players.pop(best_match_id)
                    unmatched_detections.remove(det_idx)
        
        # Create new players for unmatched detections
        for det_idx in unmatched_detections:
            new_player_id = self.next_player_id
            self.next_player_id += 1
            matched_players[new_player_id] = player_detections[det_idx]
        
        return matched_players
    
    def _hungarian_matching(self, cost_matrix: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
        """
        Perform Hungarian algorithm for optimal assignment
        
        Args:
            cost_matrix: Cost matrix for matching
            max_cost: Maximum allowed cost for a match
            
        Returns:
            List of (detection_idx, player_idx) matches
        """
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            matches = []
            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] <= max_cost:
                    matches.append((r, c))
            
            return matches
        except ImportError:
            # Fallback to greedy matching if scipy not available
            return self._greedy_matching(cost_matrix, max_cost)
    
    def _greedy_matching(self, cost_matrix: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
        """
        Greedy matching algorithm as fallback
        """
        matches = []
        used_rows = set()
        used_cols = set()
        
        # Get all possible matches and sort by cost
        possible_matches = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] <= max_cost:
                    possible_matches.append((cost_matrix[i, j], i, j))
        
        possible_matches.sort()
        
        for cost, i, j in possible_matches:
            if i not in used_rows and j not in used_cols:
                matches.append((i, j))
                used_rows.add(i)
                used_cols.add(j)
        
        return matches
    
    def update_players(self, matched_players: Dict[int, Dict], frame: np.ndarray):
        """
        Update player tracking data with current frame matches
        
        Args:
            matched_players: Dictionary of matched players
            frame: Current video frame
        """
        current_frame_players = set(matched_players.keys())
        
        # Update existing players
        for player_id, detection in matched_players.items():
            features = self.extract_features(frame, detection['bbox'])
            
            if player_id in self.players:
                # Update existing player
                player_data = self.players[player_id]
                player_data['last_center'] = detection['center']
                player_data['last_bbox'] = detection['bbox']
                player_data['confidence'] = detection['confidence']
                player_data['disappeared_count'] = 0
                player_data['trajectory'].append(detection['center'])
                player_data['feature_history'].append(features)
                
                # Update average features
                if len(player_data['feature_history']) > self.feature_history_size:
                    player_data['feature_history'].popleft()
                player_data['avg_features'] = np.mean(player_data['feature_history'], axis=0)
                
            else:
                # Create new player
                self.players[player_id] = {
                    'first_seen_frame': self.frame_count,
                    'last_center': detection['center'],
                    'last_bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'disappeared_count': 0,
                    'trajectory': deque([detection['center']], maxlen=50),
                    'feature_history': deque([features], maxlen=self.feature_history_size),
                    'avg_features': features
                }
        
        # Handle disappeared players
        for player_id in list(self.players.keys()):
            if player_id not in current_frame_players:
                self.players[player_id]['disappeared_count'] += 1
                
                if self.players[player_id]['disappeared_count'] > self.max_disappeared:
                    # Move to disappeared players for potential re-identification
                    self.disappeared_players[player_id] = self.players.pop(player_id)
                    
                    # Keep only recent disappearances
                    if len(self.disappeared_players) > 20:
                        oldest_id = min(self.disappeared_players.keys())
                        del self.disappeared_players[oldest_id]
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for player tracking
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, tracking_info)
        """
        start_time = time.time()
        
        # Detect objects in frame
        detections = self.detect_objects(frame)
        
        # Match detections to players
        matched_players = self.match_detections_to_players(detections, frame)
        
        # Update player tracking data
        self.update_players(matched_players, frame)
        
        # Create annotated frame
        annotated_frame = self.draw_annotations(frame.copy(), matched_players)
        
        # Update metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.detection_counts.append(len(matched_players))
        
        self.frame_count += 1
        
        tracking_info = {
            'frame_number': self.frame_count,
            'active_players': len(self.players),
            'disappeared_players': len(self.disappeared_players),
            'processing_time': processing_time,
            'player_positions': {pid: data['last_center'] for pid, data in self.players.items()}
        }
        
        return annotated_frame, tracking_info
    
    def draw_annotations(self, frame: np.ndarray, matched_players: Dict[int, Dict]) -> np.ndarray:
        """
        Draw tracking annotations on the frame
        
        Args:
            frame: Input frame
            matched_players: Dictionary of matched players
            
        Returns:
            Annotated frame
        """
        # Define colors for different players
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (173, 255, 47), (30, 144, 255), (220, 20, 60)
        ]
        
        for player_id, detection in matched_players.items():
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Choose color based on player ID
            color = colors[player_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID and confidence
            label = f"Player {player_id}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center = (int(detection['center'][0]), int(detection['center'][1]))
            cv2.circle(frame, center, 3, color, -1)
            
            # Draw trajectory if available
            if player_id in self.players and len(self.players[player_id]['trajectory']) > 1:
                trajectory = list(self.players[player_id]['trajectory'])
                for i in range(1, len(trajectory)):
                    pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                    pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    cv2.line(frame, pt1, pt2, color, 1)
        
        # Add frame info
        info_text = f"Frame: {self.frame_count} | Active Players: {len(self.players)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the tracking system
        
        Returns:
            Dictionary containing performance statistics
        """
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'avg_detections_per_frame': np.mean(self.detection_counts),
            'total_frames_processed': len(self.processing_times),
            'fps': 1.0 / np.mean(self.processing_times),
            'total_unique_players': self.next_player_id - 1
        }
    
    def save_tracking_results(self, output_path: str):
        """
        Save tracking results for analysis
        
        Args:
            output_path: Path to save the results
        """
        results = {
            'players': dict(self.players),
            'disappeared_players': dict(self.disappeared_players),
            'performance_metrics': self.get_performance_metrics(),
            'frame_count': self.frame_count
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)


class FeatureExtractor:
    """
    Feature extraction for player re-identification
    Uses color histograms and basic texture features
    """
    
    def __init__(self):
        self.feature_dim = 768  # Total feature dimension
        
    def extract(self, player_crop: np.ndarray) -> np.ndarray:
        """
        Extract features from player image crop
        
        Args:
            player_crop: Cropped player image
            
        Returns:
            Feature vector
        """
        if player_crop.size == 0:
            return np.zeros(self.feature_dim)
        
        # Resize to standard size
        crop_resized = cv2.resize(player_crop, (64, 128))
        
        # Color histogram features
        color_hist = self._extract_color_histogram(crop_resized)
        
        # Texture features
        texture_features = self._extract_texture_features(crop_resized)
        
        # Spatial features
        spatial_features = self._extract_spatial_features(crop_resized)
        
        # Combine all features
        features = np.concatenate([color_hist, texture_features, spatial_features])
        
        # Normalize features
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
        
        return features
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Flatten and normalize
        color_hist = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
        return color_hist / (np.sum(color_hist) + 1e-7)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using gradients"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Histogram of oriented gradients (simplified)
        mag_hist = np.histogram(magnitude.flatten(), bins=50)[0]
        dir_hist = np.histogram(direction.flatten(), bins=36)[0]
        
        texture_features = np.concatenate([mag_hist, dir_hist])
        return texture_features / (np.sum(texture_features) + 1e-7)
    
    def _extract_spatial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial layout features"""
        h, w = image.shape[:2]
        
        # Divide image into regions and extract basic statistics
        regions = []
        for i in range(0, h, h//4):
            for j in range(0, w, w//2):
                region = image[i:i+h//4, j:j+w//2]
                if region.size > 0:
                    # Basic color statistics per region
                    mean_color = np.mean(region.reshape(-1, 3), axis=0)
                    std_color = np.std(region.reshape(-1, 3), axis=0)
                    regions.extend(mean_color)
                    regions.extend(std_color)
        
        # Pad to fixed size
        spatial_features = np.array(regions)
        if len(spatial_features) < 48:
            spatial_features = np.pad(spatial_features, (0, 48 - len(spatial_features)))
        else:
            spatial_features = spatial_features[:48]
        
        return spatial_features 