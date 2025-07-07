"""
Audit logging functionality for the PDF Redactor Tool
"""

import json
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from config import AUDIT_SETTINGS


@dataclass
class RedactionEntry:
    """Data class for individual redaction entries"""
    timestamp: str
    page_number: int
    content_type: str  # 'text', 'logo', 'image'
    original_content: str
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    redaction_reason: str = ""
    confidence: float = 1.0
    method: str = ""  # 'text_layer', 'ocr', 'logo_detection'
    font_info: Optional[Dict[str, Any]] = None
    color_info: Optional[Dict[str, Any]] = None
    thumbnail_b64: Optional[str] = None


class AuditLogger:
    """Handles audit logging for PDF redaction operations"""
    
    def __init__(self, filename: str = ""):
        self.entries: List[RedactionEntry] = []
        self.filename = filename
        self.start_time = datetime.datetime.now()
        self.end_time: Optional[datetime.datetime] = None
        
    def add_text_redaction(self, 
                          page_number: int, 
                          original_text: str, 
                          bbox: List[float],
                          reason: str,
                          method: str = "text_layer",
                          confidence: float = 1.0,
                          font_info: Optional[Dict[str, Any]] = None,
                          color_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a text redaction entry to the audit log.
        
        Args:
            page_number: Page number where redaction occurred
            original_text: Original text that was redacted
            bbox: Bounding box coordinates [x0, y0, x1, y1]
            reason: Reason for redaction
            method: Method used for redaction
            confidence: Confidence level of the redaction
            font_info: Font information dictionary
            color_info: Color information dictionary
        """
        entry = RedactionEntry(
            timestamp=datetime.datetime.now().isoformat(),
            page_number=page_number,
            content_type="text",
            original_content=original_text,
            bbox=bbox,
            redaction_reason=reason,
            confidence=confidence,
            method=method,
            font_info=font_info,
            color_info=color_info
        )
        self.entries.append(entry)
    
    def add_logo_redaction(self,
                          page_number: int,
                          logo_description: str,
                          bbox: List[float],
                          confidence: float = 1.0) -> None:
        """
        Add a logo redaction entry to the audit log.
        
        Args:
            page_number: Page number where logo was found
            logo_description: Description of the logo
            bbox: Bounding box coordinates [x0, y0, x1, y1]
            confidence: Confidence level of logo detection
        """
        entry = RedactionEntry(
            timestamp=datetime.datetime.now().isoformat(),
            page_number=page_number,
            content_type="logo",
            original_content=logo_description,
            bbox=bbox,
            redaction_reason="Logo detection",
            confidence=confidence,
            method="logo_detection"
        )
        self.entries.append(entry)
    
    def add_image_redaction(self,
                           page_number: int,
                           image_description: str,
                           bbox: List[float],
                           reason: str,
                           confidence: float = 1.0) -> None:
        """
        Add an image redaction entry to the audit log.
        
        Args:
            page_number: Page number where image was found
            image_description: Description of the image
            bbox: Bounding box coordinates [x0, y0, x1, y1]
            reason: Reason for redaction
            confidence: Confidence level of detection
        """
        entry = RedactionEntry(
            timestamp=datetime.datetime.now().isoformat(),
            page_number=page_number,
            content_type="image",
            original_content=image_description,
            bbox=bbox,
            redaction_reason=reason,
            confidence=confidence,
            method="image_analysis"
        )
        self.entries.append(entry)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all redactions.
        
        Returns:
            Dictionary containing redaction summary
        """
        if not self.entries:
            return {
                "total_redactions": 0,
                "pages_processed": 0,
                "content_types": {},
                "redaction_reasons": {},
                "methods_used": {}
            }
        
        # Count by content type
        content_types = {}
        for entry in self.entries:
            content_types[entry.content_type] = content_types.get(entry.content_type, 0) + 1
        
        # Count by redaction reason
        redaction_reasons = {}
        for entry in self.entries:
            redaction_reasons[entry.redaction_reason] = redaction_reasons.get(entry.redaction_reason, 0) + 1
        
        # Count by method
        methods_used = {}
        for entry in self.entries:
            methods_used[entry.method] = methods_used.get(entry.method, 0) + 1
        
        # Get unique pages
        pages_processed = len(set(entry.page_number for entry in self.entries))
        
        # Calculate average confidence
        avg_confidence = sum(entry.confidence for entry in self.entries) / len(self.entries)
        
        return {
            "total_redactions": len(self.entries),
            "pages_processed": pages_processed,
            "content_types": content_types,
            "redaction_reasons": redaction_reasons,
            "methods_used": methods_used,
            "average_confidence": round(avg_confidence, 3),
            "processing_time_seconds": self.get_processing_time()
        }
    
    def get_processing_time(self) -> float:
        """
        Get the total processing time in seconds.
        
        Returns:
            Processing time in seconds
        """
        if self.end_time is None:
            self.end_time = datetime.datetime.now()
        
        return (self.end_time - self.start_time).total_seconds()
    
    def get_entries_by_page(self, page_number: int) -> List[RedactionEntry]:
        """
        Get all redaction entries for a specific page.
        
        Args:
            page_number: Page number to filter by
            
        Returns:
            List of redaction entries for the specified page
        """
        return [entry for entry in self.entries if entry.page_number == page_number]
    
    def get_entries_by_type(self, content_type: str) -> List[RedactionEntry]:
        """
        Get all redaction entries of a specific content type.
        
        Args:
            content_type: Content type to filter by ('text', 'logo', 'image')
            
        Returns:
            List of redaction entries of the specified type
        """
        return [entry for entry in self.entries if entry.content_type == content_type]
    
    def get_entries_by_reason(self, reason: str) -> List[RedactionEntry]:
        """
        Get all redaction entries with a specific reason.
        
        Args:
            reason: Redaction reason to filter by
            
        Returns:
            List of redaction entries with the specified reason
        """
        return [entry for entry in self.entries if entry.redaction_reason == reason]
    
    def export_to_json(self, output_path: str) -> bool:
        """
        Export audit log to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.end_time = datetime.datetime.now()
            
            audit_data = {
                "metadata": {
                    "filename": self.filename,
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat(),
                    "processing_time_seconds": self.get_processing_time(),
                    "total_entries": len(self.entries)
                },
                "summary": self.get_summary(),
                "entries": [asdict(entry) for entry in self.entries]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting audit log: {e}")
            return False
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        Export audit log to CSV file.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if not self.entries:
                    return True
                
                # Get fieldnames from the first entry
                fieldnames = asdict(self.entries[0]).keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for entry in self.entries:
                    # Convert nested dictionaries to strings for CSV
                    entry_dict = asdict(entry)
                    if entry_dict.get('font_info'):
                        entry_dict['font_info'] = json.dumps(entry_dict['font_info'])
                    if entry_dict.get('color_info'):
                        entry_dict['color_info'] = json.dumps(entry_dict['color_info'])
                    if entry_dict.get('bbox'):
                        entry_dict['bbox'] = json.dumps(entry_dict['bbox'])
                    
                    writer.writerow(entry_dict)
            
            return True
            
        except Exception as e:
            print(f"Error exporting audit log to CSV: {e}")
            return False
    
    def print_summary(self) -> None:
        """
        Print a formatted summary of the audit log to console.
        """
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("AUDIT LOG SUMMARY")
        print("="*50)
        print(f"Filename: {self.filename}")
        print(f"Total Redactions: {summary['total_redactions']}")
        print(f"Pages Processed: {summary['pages_processed']}")
        print(f"Processing Time: {summary['processing_time_seconds']:.2f} seconds")
        print(f"Average Confidence: {summary['average_confidence']:.3f}")
        
        if summary['content_types']:
            print("\nContent Types:")
            for content_type, count in summary['content_types'].items():
                print(f"  {content_type}: {count}")
        
        if summary['redaction_reasons']:
            print("\nRedaction Reasons:")
            for reason, count in summary['redaction_reasons'].items():
                print(f"  {reason}: {count}")
        
        if summary['methods_used']:
            print("\nMethods Used:")
            for method, count in summary['methods_used'].items():
                print(f"  {method}: {count}")
        
        print("="*50)
    
    def clear(self) -> None:
        """Clear all audit entries."""
        self.entries.clear()
        self.start_time = datetime.datetime.now()
        self.end_time = None 