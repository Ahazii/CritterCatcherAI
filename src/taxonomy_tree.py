"""
Hierarchical taxonomy tree for species classification.
Manages the tree structure from YOLO classes down to fine-grained species.
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TaxonomyNode:
    """A node in the taxonomy tree representing a classification level."""
    
    def __init__(
        self,
        id: str,
        name: str,
        level: str,  # 'yolo', 'species', 'subspecies', etc.
        parent_id: Optional[str] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.75,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a taxonomy node.
        
        Args:
            id: Unique identifier for this node
            name: Display name (e.g., "hedgehog", "goldfinch")
            level: Classification level ('yolo', 'species', 'subspecies', etc.)
            parent_id: ID of parent node (None for root/YOLO nodes)
            model_path: Path to trained classifier model (None for YOLO classes)
            confidence_threshold: Minimum confidence for this classifier
            enabled: Whether this node is active
            metadata: Additional node metadata (training stats, etc.)
        """
        self.id = id
        self.name = name
        self.level = level
        self.parent_id = parent_id
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        self.metadata = metadata or {}
        self.children: List[TaxonomyNode] = []
    
    def add_child(self, child: 'TaxonomyNode'):
        """Add a child node."""
        self.children.append(child)
        child.parent_id = self.id
    
    def remove_child(self, child_id: str):
        """Remove a child node by ID."""
        self.children = [c for c in self.children if c.id != child_id]
    
    def get_child(self, child_id: str) -> Optional['TaxonomyNode']:
        """Get child node by ID."""
        for child in self.children:
            if child.id == child_id:
                return child
        return None
    
    def get_path(self) -> List[str]:
        """Get path from root to this node as list of names."""
        # This will be filled in by TaxonomyTree.get_node_path()
        return [self.name]
    
    def has_model(self) -> bool:
        """Check if this node has a trained model."""
        if self.model_path:
            return Path(self.model_path).exists()
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level,
            "parent_id": self.parent_id,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "enabled": self.enabled,
            "has_model": self.has_model(),
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaxonomyNode':
        """Create node from dictionary representation."""
        node = cls(
            id=data["id"],
            name=data["name"],
            level=data["level"],
            parent_id=data.get("parent_id"),
            model_path=data.get("model_path"),
            confidence_threshold=data.get("confidence_threshold", 0.75),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {})
        )
        
        # Recursively create children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            node.add_child(child)
        
        return node


class TaxonomyTree:
    """Manages the hierarchical taxonomy tree structure."""
    
    def __init__(self, yolo_classes: List[str]):
        """
        Initialize taxonomy tree with YOLO classes as root nodes.
        
        Args:
            yolo_classes: List of YOLO COCO class names
        """
        self.roots: Dict[str, TaxonomyNode] = {}
        self._node_index: Dict[str, TaxonomyNode] = {}
        
        # Create root nodes for each YOLO class (disabled by default)
        for yolo_class in yolo_classes:
            node = TaxonomyNode(
                id=f"yolo_{yolo_class}",
                name=yolo_class,
                level="yolo",
                parent_id=None,
                model_path=None,  # YOLO doesn't need custom model
                confidence_threshold=0.0,  # Inherited from main config
                enabled=False  # Default to OFF - user must enable
            )
            self.roots[node.id] = node
            self._node_index[node.id] = node
        
        logger.info(f"Initialized taxonomy tree with {len(self.roots)} YOLO root classes")
    
    def add_node(
        self,
        name: str,
        parent_id: str,
        level: str,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.75,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaxonomyNode:
        """
        Add a new node to the tree.
        
        Args:
            name: Species/subspecies name
            parent_id: ID of parent node
            level: Classification level
            model_path: Path to trained model
            confidence_threshold: Confidence threshold
            metadata: Additional metadata
            
        Returns:
            Created node
            
        Raises:
            ValueError: If parent node not found
        """
        parent = self.get_node(parent_id)
        if not parent:
            raise ValueError(f"Parent node '{parent_id}' not found")
        
        # Generate unique ID
        node_id = self._generate_id(name, parent_id)
        
        # Create node
        node = TaxonomyNode(
            id=node_id,
            name=name,
            level=level,
            parent_id=parent_id,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            enabled=True,
            metadata=metadata or {}
        )
        
        # Add to tree
        parent.add_child(node)
        self._node_index[node_id] = node
        
        logger.info(f"Added node '{name}' ({level}) under '{parent.name}'")
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its descendants.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if removed, False if not found
        """
        if node_id.startswith("yolo_"):
            logger.warning(f"Cannot remove YOLO root node: {node_id}")
            return False
        
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return False
        
        parent = self.get_node(node.parent_id)
        if parent:
            parent.remove_child(node_id)
        
        # Remove from index (including all descendants)
        self._remove_from_index(node)
        
        logger.info(f"Removed node '{node.name}' and its descendants")
        return True
    
    def _remove_from_index(self, node: TaxonomyNode):
        """Recursively remove node and descendants from index."""
        if node.id in self._node_index:
            del self._node_index[node.id]
        
        for child in node.children:
            self._remove_from_index(child)
    
    def get_node(self, node_id: str) -> Optional[TaxonomyNode]:
        """Get node by ID."""
        return self._node_index.get(node_id)
    
    def get_node_path(self, node_id: str) -> List[str]:
        """
        Get full path from root to node as list of names.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of names from root to node (e.g., ['bird', 'finch', 'goldfinch'])
        """
        node = self.get_node(node_id)
        if not node:
            return []
        
        path = [node.name]
        current = node
        
        while current.parent_id:
            parent = self.get_node(current.parent_id)
            if not parent:
                break
            path.insert(0, parent.name)
            current = parent
        
        return path
    
    def get_classifier_chain(self, yolo_class: str) -> List[TaxonomyNode]:
        """
        Get ordered chain of classifiers to run for a YOLO detection.
        
        Args:
            yolo_class: YOLO class name (e.g., 'bird', 'cat')
            
        Returns:
            Flat list of nodes with trained models in tree traversal order
        """
        root_id = f"yolo_{yolo_class}"
        root = self.roots.get(root_id)
        
        if not root:
            return []
        
        # Collect all enabled nodes with models
        classifiers = []
        self._collect_classifiers(root, classifiers)
        return classifiers
    
    def _collect_classifiers(self, node: TaxonomyNode, result: List[TaxonomyNode]):
        """Recursively collect classifier nodes."""
        for child in node.children:
            if child.enabled and child.has_model():
                result.append(child)
            self._collect_classifiers(child, result)
    
    def _generate_id(self, name: str, parent_id: str) -> str:
        """Generate unique node ID."""
        base_id = f"{parent_id}_{name.lower().replace(' ', '_')}"
        
        # Handle collisions
        if base_id in self._node_index:
            counter = 1
            while f"{base_id}_{counter}" in self._node_index:
                counter += 1
            base_id = f"{base_id}_{counter}"
        
        return base_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Export entire tree to dictionary."""
        return {
            "roots": {root_id: root.to_dict() for root_id, root in self.roots.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], yolo_classes: List[str]) -> 'TaxonomyTree':
        """Import tree from dictionary."""
        tree = cls(yolo_classes)
        
        # Load saved tree structure
        for root_id, root_data in data.get("roots", {}).items():
            if root_id in tree.roots:
                # Replace default root with saved version
                saved_root = TaxonomyNode.from_dict(root_data)
                tree.roots[root_id] = saved_root
                
                # Rebuild node index
                tree._rebuild_index()
        
        return tree
    
    def _rebuild_index(self):
        """Rebuild the node index after loading."""
        self._node_index = {}
        for root in self.roots.values():
            self._index_node(root)
    
    def _index_node(self, node: TaxonomyNode):
        """Recursively index a node and its children."""
        self._node_index[node.id] = node
        for child in node.children:
            self._index_node(child)
    
    def save_to_file(self, file_path: Path):
        """Save tree to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved taxonomy tree to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Path, yolo_classes: List[str]) -> 'TaxonomyTree':
        """Load tree from JSON file."""
        if not file_path.exists():
            logger.info(f"Taxonomy file not found, creating new tree")
            return cls(yolo_classes)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        tree = cls.from_dict(data, yolo_classes)
        logger.info(f"Loaded taxonomy tree from {file_path}")
        return tree
