"""
Enhanced Text processing and pattern matching for the PDF Redactor Tool
Now with spaCy integration for intelligent text classification
"""

import re
import string
from typing import List, Dict, Any, Tuple, Optional
from config import (
    COMPANY_TERMS, DRAWING_PATTERNS, EMAIL_PATTERN, 
    PHONE_PATTERNS, ADDRESS_PATTERNS, TECHNICAL_PRESERVE
)
from utils import is_technical_content

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")


class TechnicalTextClassifier:
    """Enhanced text classification using spaCy for technical content detection"""
    
    def __init__(self):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self._add_technical_patterns()
            except OSError:
                print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def _add_technical_patterns(self):
        """Add custom patterns for technical content detection"""
        if not self.nlp:
            return
            
        # Add custom patterns for technical content
        ruler = self.nlp.get_pipe("entity_ruler") if "entity_ruler" in self.nlp.pipe_names else self.nlp.add_pipe("entity_ruler")
        
        patterns = [
            # Dimensions
            {"label": "DIMENSION", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["mm", "cm", "in", "ft", "m", "inch", "inches"]}}]},
            {"label": "DIMENSION", "pattern": [{"LOWER": {"IN": ["mm", "cm", "in", "ft", "m", "inch", "inches"]}}, {"LIKE_NUM": True}]},
            
            # Tolerances
            {"label": "TOLERANCE", "pattern": [{"TEXT": "±"}, {"LIKE_NUM": True}]},
            {"label": "TOLERANCE", "pattern": [{"TEXT": "+/-"}, {"LIKE_NUM": True}]},
            
            # Materials
            {"label": "MATERIAL", "pattern": [{"LOWER": {"IN": ["steel", "aluminum", "aluminium", "plastic", "copper", "brass", "titanium", "stainless"]}}]},
            
            # Technical specifications
            {"label": "TECH_SPEC", "pattern": [{"LOWER": {"IN": ["tolerance", "specification", "standard", "grade", "class"]}}]},
            
            # Part numbers (common patterns)
            {"label": "PART_NUMBER", "pattern": [{"SHAPE": "XXXX"}, {"LIKE_NUM": True}]},
            {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"[A-Z]{2,4}-\d{3,6}"}}]},
            
            # Angles and degrees
            {"label": "ANGLE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["deg", "degree", "°"]}}]},
            
            # Surface finish
            {"label": "SURFACE_FINISH", "pattern": [{"LOWER": {"IN": ["ra", "rms", "surface", "finish"]}}]},
        ]
        
        ruler.add_patterns(patterns)
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text using spaCy for technical content detection
        
        Args:
            text: Text to classify
            
        Returns:
            Classification results with confidence scores
        """
        if not self.nlp or not text.strip():
            return {"is_technical": False, "confidence": 0.0, "entities": []}
        
        doc = self.nlp(text.strip())
        
        # Count technical entities
        technical_entities = []
        for ent in doc.ents:
            if ent.label_ in ["DIMENSION", "TOLERANCE", "MATERIAL", "TECH_SPEC", "PART_NUMBER", "ANGLE", "SURFACE_FINISH"]:
                technical_entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Calculate technical content score
        technical_score = len(technical_entities) / max(len(doc), 1)
        
        # Additional technical indicators
        technical_indicators = [
            any(token.like_num for token in doc),  # Contains numbers
            any(token.text in ["±", "+/-", "°", "deg"] for token in doc),  # Technical symbols
            any(token.text.lower() in ["mm", "cm", "in", "ft", "m"] for token in doc),  # Units
            any(token.text.lower() in ["tolerance", "spec", "grade", "class"] for token in doc),  # Technical terms
        ]
        
        technical_indicator_score = sum(technical_indicators) / len(technical_indicators)
        
        # Combined confidence score
        confidence = (technical_score * 0.6) + (technical_indicator_score * 0.4)
        
        return {
            "is_technical": confidence > 0.3,
            "confidence": confidence,
            "entities": technical_entities,
            "technical_score": technical_score,
            "indicator_score": technical_indicator_score
        }


class TextProcessor:
    """Enhanced text analysis and pattern matching for redaction with spaCy integration"""
    
    def __init__(self):
        self.company_patterns = self._compile_company_patterns()
        self.drawing_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DRAWING_PATTERNS]
        self.email_pattern = re.compile(EMAIL_PATTERN, re.IGNORECASE)
        self.phone_patterns = [re.compile(pattern) for pattern in PHONE_PATTERNS]
        self.address_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in ADDRESS_PATTERNS]
        
        # Initialize spaCy classifier
        self.technical_classifier = TechnicalTextClassifier()
        
    def _compile_company_patterns(self) -> List[re.Pattern]:
        """Compile company name patterns with word boundaries"""
        patterns = []
        for term in COMPANY_TERMS:
            # Create pattern with word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    def analyze_text(self, text: str, bbox: List[float], page_number: int) -> List[Dict[str, Any]]:
        """
        Enhanced text analysis with spaCy classification for redaction candidates.
        
        Args:
            text: Text to analyze
            bbox: Bounding box coordinates [x0, y0, x1, y1]
            page_number: Page number
            
        Returns:
            List of redaction candidates with reasons and confidence
        """
        candidates = []
        
        if not text or not text.strip():
            return candidates
        
        # Use spaCy to classify text as technical content
        classification = self.technical_classifier.classify_text(text)
        
        # Skip if it's technical content that should be preserved
        if classification["is_technical"] and classification["confidence"] > 0.5:
            print(f"Skipping technical content: '{text[:50]}...' (confidence: {classification['confidence']:.2f})")
            return candidates
        
        # Also check with legacy method for backward compatibility
        if is_technical_content(text):
            return candidates
        
        # Check for company names
        company_matches = self._check_company_names(text)
        for match in company_matches:
            candidates.append({
                'text': match['text'],
                'reason': 'Company name detected',
                'confidence': 1.0,
                'bbox': bbox,
                'page': page_number,
                'type': 'company_name',
                'spacy_classification': classification
            })
        
        # Check for drawing numbers
        drawing_matches = self._check_drawing_numbers(text)
        for match in drawing_matches:
            candidates.append({
                'text': match['text'],
                'reason': 'Drawing number detected',
                'confidence': 0.9,
                'bbox': bbox,
                'page': page_number,
                'type': 'drawing_number',
                'spacy_classification': classification
            })
        
        # Check for email addresses
        email_matches = self._check_email_addresses(text)
        for match in email_matches:
            candidates.append({
                'text': match['text'],
                'reason': 'Email address detected',
                'confidence': 1.0,
                'bbox': bbox,
                'page': page_number,
                'type': 'email',
                'spacy_classification': classification
            })
        
        # Check for phone numbers
        phone_matches = self._check_phone_numbers(text)
        for match in phone_matches:
            candidates.append({
                'text': match['text'],
                'reason': 'Phone number detected',
                'confidence': 0.95,
                'bbox': bbox,
                'page': page_number,
                'type': 'phone',
                'spacy_classification': classification
            })
        
        # Check for addresses
        address_matches = self._check_addresses(text)
        for match in address_matches:
            candidates.append({
                'text': match['text'],
                'reason': 'Address detected',
                'confidence': 0.8,
                'bbox': bbox,
                'page': page_number,
                'type': 'address',
                'spacy_classification': classification
            })
        
        # Check for file paths or CAD references
        file_matches = self._check_file_references(text)
        for match in file_matches:
            candidates.append({
                'text': match['text'],
                'reason': 'File reference detected',
                'confidence': 0.85,
                'bbox': bbox,
                'page': page_number,
                'type': 'file_reference',
                'spacy_classification': classification
            })
        
        return candidates
    
    def _check_company_names(self, text: str) -> List[Dict[str, Any]]:
        """Check for company name matches"""
        matches = []
        for pattern in self.company_patterns:
            for match in pattern.finditer(text):
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        return matches
    
    def _check_drawing_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Check for drawing number patterns"""
        matches = []
        for pattern in self.drawing_patterns:
            for match in pattern.finditer(text):
                # Additional validation for drawing numbers
                matched_text = match.group()
                if self._is_likely_drawing_number(matched_text):
                    matches.append({
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end()
                    })
        return matches
    
    def _is_likely_drawing_number(self, text: str) -> bool:
        """Additional validation for drawing numbers"""
        # Skip if it's clearly a measurement or technical spec
        if re.search(r'\d+\.\d+', text):  # Decimal numbers like 0.105
            return False
        if re.search(r'±\d+', text):  # Tolerances
            return False
        if re.search(r'\d+\s*(?:mm|cm|in|ft|m|°|deg)', text):  # Measurements
            return False
        
        # Skip if it's too short (likely just a number)
        if len(text.strip()) < 4:
            return False
            
        # Skip if it contains common technical terms
        technical_terms = ['mm', 'cm', 'in', 'ft', 'm', 'deg', '°', '±', 'tolerance', 'spec']
        text_lower = text.lower()
        for term in technical_terms:
            if term in text_lower:
                return False
        
        # Skip if it's just a sequence of digits (likely a measurement)
        if re.match(r'^\d+$', text.strip()):
            return False
            
        # Skip if it contains common measurement patterns
        if re.search(r'\d+\s*[xX×]\s*\d+', text):  # Dimensions like "100 x 200"
            return False
        
        return True
    
    def _check_email_addresses(self, text: str) -> List[Dict[str, Any]]:
        """Check for email addresses"""
        matches = []
        for match in self.email_pattern.finditer(text):
            matches.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        return matches
    
    def _check_phone_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Check for phone numbers"""
        matches = []
        for pattern in self.phone_patterns:
            for match in pattern.finditer(text):
                # Additional validation for phone numbers
                matched_text = match.group()
                if self._is_likely_phone_number(matched_text):
                    matches.append({
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end()
                    })
        return matches
    
    def _is_likely_phone_number(self, text: str) -> bool:
        """Additional validation for phone numbers"""
        # Remove common separators
        cleaned = re.sub(r'[-.()\s]', '', text)
        
        # Should be exactly 10 digits for US phone numbers
        if len(cleaned) == 10 and cleaned.isdigit():
            return True
        
        # Allow for country codes (11 digits starting with 1)
        if len(cleaned) == 11 and cleaned.startswith('1') and cleaned[1:].isdigit():
            return True
        
        return False
    
    def _check_addresses(self, text: str) -> List[Dict[str, Any]]:
        """Check for addresses"""
        matches = []
        for pattern in self.address_patterns:
            for match in pattern.finditer(text):
                matched_text = match.group()
                if self._is_likely_address(matched_text):
                    matches.append({
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end()
                    })
        return matches
    
    def _is_likely_address(self, text: str) -> bool:
        """Additional validation for addresses"""
        # Should contain street type and be reasonably long
        if len(text.split()) < 3:
            return False
        
        # Should contain a number
        if not re.search(r'\d+', text):
            return False
        
        # Skip if it contains technical terms or instructions
        technical_terms = ['per', 'vendor', 'request', 'spec', 'tolerance', 'mm', 'cm', 'in', 'ft', 'm']
        text_lower = text.lower()
        for term in technical_terms:
            if term in text_lower:
                return False
        
        # Should contain common address indicators
        address_indicators = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd', 'drive', 'dr', 'lane', 'ln', 'court', 'ct', 'place', 'pl', 'way', 'terrace', 'ter']
        has_address_indicator = any(indicator in text_lower for indicator in address_indicators)
        
        # Should contain a state abbreviation or city pattern
        has_state_pattern = bool(re.search(r'\b[A-Z]{2}\b', text))  # State abbreviation like NY, CA
        
        return has_address_indicator or has_state_pattern
    
    def _check_file_references(self, text: str) -> List[Dict[str, Any]]:
        """Check for file references and CAD file names"""
        # Pattern for file references
        file_patterns = [
            r'\b[A-Za-z0-9_-]+\.(?:cad|dwg|dxf|pdf|jpg|png|tif)\b',
            r'\b[A-Za-z0-9_-]+_[A-Za-z0-9_-]+\.(?:cad|dwg|dxf)\b',
            r'\b[A-Za-z0-9_-]+_\d{2}_[A-Za-z0-9_-]+\.(?:cad|dwg|dxf)\b'
        ]
        
        matches = []
        for pattern in file_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group()
                # Skip if it's clearly a technical reference
                if not self._is_technical_file_reference(matched_text):
                    matches.append({
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end()
                    })
        return matches
    
    def _is_technical_file_reference(self, text: str) -> bool:
        """Check if file reference is technical (should be preserved)"""
        # Technical file patterns that should be preserved
        technical_patterns = [
            r'\.(?:jpg|png|tif|bmp)$',  # Image files
            r'technical_',  # Technical documentation
            r'spec_',  # Specifications
            r'std_',  # Standards
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def get_text_blocks_for_redaction(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process text blocks and identify those that need redaction.
        
        Args:
            text_blocks: List of text block dictionaries from PDF
            
        Returns:
            List of text blocks that should be redacted
        """
        redaction_candidates = []
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
            
            # Analyze the text block
            candidates = self.analyze_text(
                text=text,
                bbox=block.get('bbox', []),
                page_number=block.get('page', 1)
            )
            
            if candidates:
                # Add block information to candidates
                for candidate in candidates:
                    candidate.update({
                        'font': block.get('font'),
                        'size': block.get('size'),
                        'color': block.get('color'),
                        'flags': block.get('flags')
                    })
                
                redaction_candidates.extend(candidates)
        
        return redaction_candidates
    
    def merge_overlapping_redactions(self, redactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping redaction areas to avoid double-redaction.
        
        Args:
            redactions: List of redaction candidates
            
        Returns:
            List of merged redactions
        """
        if not redactions:
            return []
        
        # Sort by page, then by y-coordinate
        sorted_redactions = sorted(redactions, key=lambda x: (x['page'], x['bbox'][1]))
        
        merged = []
        current_group = [sorted_redactions[0]]
        
        for redaction in sorted_redactions[1:]:
            if self._redactions_overlap(current_group[-1], redaction):
                current_group.append(redaction)
            else:
                # Merge current group
                merged.append(self._merge_redaction_group(current_group))
                current_group = [redaction]
        
        # Don't forget the last group
        if current_group:
            merged.append(self._merge_redaction_group(current_group))
        
        return merged
    
    def _redactions_overlap(self, red1: Dict[str, Any], red2: Dict[str, Any]) -> bool:
        """Check if two redactions overlap"""
        if red1['page'] != red2['page']:
            return False
        
        bbox1 = red1['bbox']
        bbox2 = red2['bbox']
        
        # Check for overlap
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    
    def _merge_redaction_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of overlapping redactions"""
        if len(group) == 1:
            return group[0]
        
        # Merge bounding boxes
        bboxes = [r['bbox'] for r in group]
        merged_bbox = [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes)
        ]
        
        # Combine reasons
        reasons = list(set(r['reason'] for r in group))
        combined_reason = '; '.join(reasons)
        
        # Use highest confidence
        max_confidence = max(r['confidence'] for r in group)
        
        # Combine text (for audit purposes)
        combined_text = ' '.join(r['text'] for r in group)
        
        return {
            'text': combined_text,
            'reason': combined_reason,
            'confidence': max_confidence,
            'bbox': merged_bbox,
            'page': group[0]['page'],
            'type': 'merged',
            'font': group[0].get('font'),
            'size': group[0].get('size'),
            'color': group[0].get('color'),
            'flags': group[0].get('flags')
        } 