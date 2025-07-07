"""
Logo detection functionality for the PDF Redactor Tool
"""

import cv2  # type: ignore
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config import LOGO_DETECTION
from utils import preprocess_image_for_ocr


class LogoDetector:
    """Detects logos and company branding in images"""
    
    def __init__(self):
        self.min_confidence = LOGO_DETECTION['min_confidence']
        self.template_threshold = LOGO_DETECTION['template_matching_threshold']
        self.edge_sensitivity = LOGO_DETECTION['edge_detection_sensitivity']
        self.min_logo_size = LOGO_DETECTION['min_logo_size']
        self.max_logo_size = LOGO_DETECTION['max_logo_size']
        
        # Common logo characteristics - MUCH MORE CONSERVATIVE
        self.logo_characteristics = {
            'min_aspect_ratio': 0.5,  # More restrictive aspect ratio
            'max_aspect_ratio': 3.0,  # More restrictive aspect ratio
            'min_area': 5000,  # Much larger minimum area (was 1000)
            'max_area': 50000,  # Smaller maximum area (was 100000)
        }
        
        # Increase minimum confidence significantly
        self.min_confidence = 0.85  # Was 0.7, now much higher
    
    def detect_logos(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect logos in an image using multiple detection methods.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected logos with bounding boxes and confidence scores
        """
        logos = []
        
        # Method 1: Template matching (if we have logo templates)
        template_logos = self._template_matching(image)
        logos.extend(template_logos)
        
        # Method 2: Contour-based detection
        contour_logos = self._contour_based_detection(image)
        logos.extend(contour_logos)
        
        # Method 3: Edge-based detection
        edge_logos = self._edge_based_detection(image)
        logos.extend(edge_logos)
        
        # Method 4: Color-based detection
        color_logos = self._color_based_detection(image)
        logos.extend(color_logos)
        
        # Remove duplicates and merge overlapping detections
        merged_logos = self._merge_overlapping_detections(logos)
        
        # Filter by confidence and size
        filtered_logos = self._filter_detections(merged_logos)
        
        # Limit the number of detections to prevent overwhelming the system
        # Sort by confidence and take only the top 10 most confident detections
        filtered_logos.sort(key=lambda x: x['confidence'], reverse=True)
        filtered_logos = filtered_logos[:10]
        
        return filtered_logos
    
    def _template_matching(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect logos using template matching.
        This method requires pre-defined logo templates.
        
        Args:
            image: Input image
            
        Returns:
            List of detected logos
        """
        logos = []
        
        # Convert to grayscale for template matching
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # TODO: Load logo templates from a database or configuration
        # For now, we'll use some common logo characteristics
        
        # Example: Look for rectangular regions with high contrast
        # This is a simplified approach - in practice, you'd have actual logo templates
        
        # Find regions with high contrast (potential logos)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        
        if variance > 100:  # High variance indicates potential logo regions
            # Find contours in the image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.logo_characteristics['min_area'] and area < self.logo_characteristics['max_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if (self.logo_characteristics['min_aspect_ratio'] <= aspect_ratio <= 
                        self.logo_characteristics['max_aspect_ratio']):
                        
                        # Calculate confidence based on contour properties
                        confidence = self._calculate_contour_confidence(contour, gray)
                        
                        if confidence > self.min_confidence:
                            logos.append({
                                'bbox': [x, y, x + w, y + h],
                                'confidence': confidence,
                                'method': 'template_matching',
                                'area': area,
                                'aspect_ratio': aspect_ratio
                            })
        
        return logos
    
    def _contour_based_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect logos using contour analysis.
        
        Args:
            image: Input image
            
        Returns:
            List of detected logos
        """
        logos = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first) and limit processing
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]  # Only process top 50 largest contours
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - much more restrictive
            if area < self.logo_characteristics['min_area'] or area > self.logo_characteristics['max_area']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter by aspect ratio
            if not (self.logo_characteristics['min_aspect_ratio'] <= aspect_ratio <= 
                   self.logo_characteristics['max_aspect_ratio']):
                continue
            
            # Additional filtering: check if the region is too close to the edges
            img_height, img_width = gray.shape
            margin = 20
            if (x < margin or y < margin or 
                x + w > img_width - margin or y + h > img_height - margin):
                continue
            
            # Calculate confidence based on contour properties
            confidence = self._calculate_contour_confidence(contour, gray)
            
            # Much higher confidence threshold
            if confidence > 0.9:  # Increased from min_confidence
                logos.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': confidence,
                    'method': 'contour_based',
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        return logos
    
    def _edge_based_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect logos using edge detection.
        
        Args:
            image: Input image
            
        Returns:
            List of detected logos
        """
        logos = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours in edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.logo_characteristics['min_area'] or area > self.logo_characteristics['max_area']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter by aspect ratio
            if not (self.logo_characteristics['min_aspect_ratio'] <= aspect_ratio <= 
                   self.logo_characteristics['max_aspect_ratio']):
                continue
            
            # Calculate edge density (ratio of edge pixels to total area)
            roi = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi > 0) / (w * h)
            
            # Logos typically have high edge density
            if edge_density > self.edge_sensitivity:
                confidence = min(edge_density * 2, 1.0)  # Scale to 0-1
                
                if confidence > self.min_confidence:
                    logos.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'method': 'edge_based',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'edge_density': edge_density
                    })
        
        return logos
    
    def _color_based_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect logos using color analysis.
        
        Args:
            image: Input image
            
        Returns:
            List of detected logos
        """
        logos = []
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Look for regions with consistent color (potential logos)
        # This is a simplified approach - more sophisticated color analysis could be implemented
        
        # Calculate color variance in different regions
        height, width = image.shape[:2]
        step = 50  # Analyze every 50x50 pixel region
        
        for y in range(0, height - step, step):
            for x in range(0, width - step, step):
                roi = image[y:y+step, x:x+step]
                
                if roi.size == 0:
                    continue
                
                # Calculate color variance
                color_variance = np.var(roi)
                
                # Logos often have low color variance (consistent colors)
                if color_variance < 1000:  # Threshold for low variance
                    # Check if this region has reasonable size
                    if step >= self.min_logo_size[0] and step >= self.min_logo_size[1]:
                        confidence = max(0.0, 1.0 - (float(color_variance) / 1000))
                        
                        if confidence > self.min_confidence:
                            logos.append({
                                'bbox': [x, y, x + step, y + step],
                                'confidence': confidence,
                                'method': 'color_based',
                                'area': step * step,
                                'aspect_ratio': 1.0,
                                'color_variance': color_variance
                            })
        
        return logos
    
    def _calculate_contour_confidence(self, contour: np.ndarray, gray_image: np.ndarray) -> float:
        """
        Calculate confidence score for a contour based on various properties.
        
        Args:
            contour: Contour to analyze
            gray_image: Grayscale image
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity (logos often have regular shapes)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate contrast in the region
        roi = gray_image[y:y+h, x:x+w]
        if roi.size > 0:
            contrast = np.std(roi)
        else:
            contrast = 0
        
        # Combine factors into confidence score
        confidence = 0.0
        
        # Area factor (prefer medium-sized regions)
        if self.logo_characteristics['min_area'] <= area <= self.logo_characteristics['max_area']:
            area_factor = 1.0
        else:
            area_factor = 0.0
        
        # Aspect ratio factor
        if self.logo_characteristics['min_aspect_ratio'] <= aspect_ratio <= self.logo_characteristics['max_aspect_ratio']:
            aspect_factor = 1.0
        else:
            aspect_factor = 0.0
        
        # Circularity factor (prefer regular shapes)
        circularity_factor = min(circularity * 2, 1.0)
        
        # Solidity factor (prefer solid shapes)
        solidity_factor = solidity
        
        # Contrast factor (prefer high contrast)
        contrast_factor = min(float(contrast) / 50, 1.0)
        
        # Weighted combination
        confidence = (
            0.2 * area_factor +
            0.2 * aspect_factor +
            0.2 * circularity_factor +
            0.2 * solidity_factor +
            0.2 * contrast_factor
        )
        
        return confidence
    
    def _merge_overlapping_detections(self, logos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping logo detections.
        
        Args:
            logos: List of logo detections
            
        Returns:
            List of merged detections
        """
        if not logos:
            return []
        
        # Sort by confidence (highest first)
        sorted_logos = sorted(logos, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, logo in enumerate(sorted_logos):
            if i in used:
                continue
            
            current_group = [logo]
            used.add(i)
            
            # Find overlapping detections
            for j, other_logo in enumerate(sorted_logos[i+1:], i+1):
                if j in used:
                    continue
                
                if self._detections_overlap(logo, other_logo):
                    current_group.append(other_logo)
                    used.add(j)
            
            # Merge the group
            if len(current_group) == 1:
                merged.append(logo)
            else:
                merged.append(self._merge_detection_group(current_group))
        
        return merged
    
    def _detections_overlap(self, logo1: Dict[str, Any], logo2: Dict[str, Any]) -> bool:
        """
        Check if two logo detections overlap.
        
        Args:
            logo1: First logo detection
            logo2: Second logo detection
            
        Returns:
            True if detections overlap
        """
        bbox1 = logo1['bbox']
        bbox2 = logo2['bbox']
        
        # Calculate intersection over union (IoU)
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou > 0.3  # Threshold for overlap
    
    def _merge_detection_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a group of overlapping detections.
        
        Args:
            group: List of overlapping detections
            
        Returns:
            Merged detection
        """
        # Merge bounding boxes
        bboxes = [logo['bbox'] for logo in group]
        merged_bbox = [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes)
        ]
        
        # Use highest confidence
        max_confidence = max(logo['confidence'] for logo in group)
        
        # Combine methods
        methods = list(set(logo['method'] for logo in group))
        combined_method = '+'.join(methods)
        
        return {
            'bbox': merged_bbox,
            'confidence': max_confidence,
            'method': combined_method,
            'area': (merged_bbox[2] - merged_bbox[0]) * (merged_bbox[3] - merged_bbox[1]),
            'aspect_ratio': (merged_bbox[2] - merged_bbox[0]) / (merged_bbox[3] - merged_bbox[1])
        }
    
    def _filter_detections(self, logos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections based on confidence and size criteria.
        
        Args:
            logos: List of logo detections
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for logo in logos:
            # Check confidence
            if logo['confidence'] < self.min_confidence:
                continue
            
            # Check size
            width = logo['bbox'][2] - logo['bbox'][0]
            height = logo['bbox'][3] - logo['bbox'][1]
            
            if (width < self.min_logo_size[0] or height < self.min_logo_size[1] or
                width > self.max_logo_size[0] or height > self.max_logo_size[1]):
                continue
            
            filtered.append(logo)
        
        return filtered 