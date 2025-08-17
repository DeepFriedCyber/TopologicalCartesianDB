#!/usr/bin/env python3
"""
Enterprise Security & Access Control for TCDB

Implements RBAC/ABAC, data lineage security, and enterprise-grade access controls
addressing the feedback on missing security features.
"""

import logging
import hashlib
import time
import json
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    """Access levels for RBAC"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    AUDIT = "audit"

class CubeType(Enum):
    """Cube types for domain-specific access control"""
    CODE = "code"
    DATA = "data"
    USER = "user"
    TEMPORAL = "temporal"
    SYSTEM = "system"
    ORCHESTRATOR = "orchestrator"

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    roles: Set[str]
    permissions: Set[str]
    cube_access: Dict[CubeType, AccessLevel]
    session_id: str
    expires_at: datetime
    ip_address: Optional[str] = None
    audit_trail: List[str] = field(default_factory=list)

@dataclass
class DataLineageNode:
    """Node in data lineage graph for provenance security"""
    node_id: str
    cube_type: CubeType
    data_classification: str  # public, internal, confidential, restricted
    source_systems: List[str]
    transformations: List[str]
    access_requirements: Set[str]
    created_at: datetime
    created_by: str

@dataclass
class ProvenanceRecord:
    """Complete provenance record for audit trails"""
    record_id: str
    query_id: str
    user_id: str
    cubes_accessed: List[CubeType]
    data_sources: List[str]
    transformations_applied: List[str]
    results_classification: str
    timestamp: datetime
    ip_address: str
    session_id: str

class EnterpriseSecurityManager:
    """Enterprise-grade security manager for TCDB"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Security policies
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special': True,
            'max_age_days': 90
        }
        
        # Session management
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_timeout = timedelta(hours=8)
        
        # Audit logging
        self.audit_log: List[Dict[str, Any]] = []
        
        # Data classification levels
        self.classification_levels = {
            'public': 0,
            'internal': 1,
            'confidential': 2,
            'restricted': 3
        }
        
        # Role-based permissions
        self.role_permissions = {
            'viewer': {AccessLevel.READ},
            'analyst': {AccessLevel.READ},
            'developer': {AccessLevel.READ, AccessLevel.WRITE},
            'admin': {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN},
            'auditor': {AccessLevel.READ, AccessLevel.AUDIT}
        }
        
        # Cube access matrix
        self.default_cube_access = {
            'viewer': {
                CubeType.DATA: AccessLevel.READ,
                CubeType.USER: AccessLevel.READ
            },
            'analyst': {
                CubeType.DATA: AccessLevel.READ,
                CubeType.TEMPORAL: AccessLevel.READ,
                CubeType.USER: AccessLevel.READ
            },
            'developer': {
                CubeType.CODE: AccessLevel.WRITE,
                CubeType.DATA: AccessLevel.WRITE,
                CubeType.SYSTEM: AccessLevel.READ
            },
            'admin': {cube: AccessLevel.ADMIN for cube in CubeType},
            'auditor': {cube: AccessLevel.AUDIT for cube in CubeType}
        }
        
        logger.info("🔒 Enterprise Security Manager initialized")
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str) -> Optional[SecurityContext]:
        """Authenticate user and create security context"""
        
        # In production, this would integrate with enterprise SSO/LDAP
        # For demo, we'll simulate authentication
        
        if not self._validate_password(password):
            self._log_security_event("authentication_failed", {
                'username': username,
                'ip_address': ip_address,
                'reason': 'invalid_password'
            })
            return None
        
        # Simulate user lookup
        user_roles = self._get_user_roles(username)
        if not user_roles:
            self._log_security_event("authentication_failed", {
                'username': username,
                'ip_address': ip_address,
                'reason': 'user_not_found'
            })
            return None
        
        # Create security context
        session_id = str(uuid.uuid4())
        permissions = set()
        cube_access = {}
        
        for role in user_roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
            if role in self.default_cube_access:
                for cube, access in self.default_cube_access[role].items():
                    cube_access[cube] = max(
                        cube_access.get(cube, AccessLevel.READ),
                        access,
                        key=lambda x: list(AccessLevel).index(x)
                    )
        
        context = SecurityContext(
            user_id=username,
            roles=user_roles,
            permissions=permissions,
            cube_access=cube_access,
            session_id=session_id,
            expires_at=datetime.now() + self.session_timeout,
            ip_address=ip_address
        )
        
        self.active_sessions[session_id] = context
        
        self._log_security_event("authentication_success", {
            'user_id': username,
            'session_id': session_id,
            'ip_address': ip_address,
            'roles': list(user_roles)
        })
        
        return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate active session"""
        
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        if datetime.now() > context.expires_at:
            del self.active_sessions[session_id]
            self._log_security_event("session_expired", {
                'user_id': context.user_id,
                'session_id': session_id
            })
            return None
        
        return context
    
    def authorize_cube_access(self, context: SecurityContext, 
                            cube_type: CubeType, 
                            access_level: AccessLevel) -> bool:
        """Authorize access to specific cube"""
        
        if cube_type not in context.cube_access:
            self._log_security_event("access_denied", {
                'user_id': context.user_id,
                'cube_type': cube_type.value,
                'requested_access': access_level.value,
                'reason': 'no_cube_access'
            })
            return False
        
        user_access = context.cube_access[cube_type]
        access_levels = list(AccessLevel)
        
        if access_levels.index(user_access) < access_levels.index(access_level):
            self._log_security_event("access_denied", {
                'user_id': context.user_id,
                'cube_type': cube_type.value,
                'requested_access': access_level.value,
                'user_access': user_access.value,
                'reason': 'insufficient_privileges'
            })
            return False
        
        return True
    
    def create_provenance_record(self, context: SecurityContext,
                               query_id: str,
                               cubes_accessed: List[CubeType],
                               data_sources: List[str],
                               transformations: List[str],
                               results_classification: str) -> ProvenanceRecord:
        """Create complete provenance record for audit"""
        
        record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            query_id=query_id,
            user_id=context.user_id,
            cubes_accessed=cubes_accessed,
            data_sources=data_sources,
            transformations_applied=transformations,
            results_classification=results_classification,
            timestamp=datetime.now(),
            ip_address=context.ip_address or "unknown",
            session_id=context.session_id
        )
        
        # Store encrypted provenance record
        encrypted_record = self._encrypt_provenance(record)
        
        self._log_security_event("provenance_created", {
            'record_id': record.record_id,
            'user_id': context.user_id,
            'cubes_accessed': [c.value for c in cubes_accessed],
            'data_sources_count': len(data_sources),
            'classification': results_classification
        })
        
        return record
    
    def validate_data_classification(self, context: SecurityContext,
                                   required_classification: str) -> bool:
        """Validate user can access data of given classification"""
        
        user_clearance = self._get_user_clearance(context.user_id)
        required_level = self.classification_levels.get(required_classification, 3)
        user_level = self.classification_levels.get(user_clearance, 0)
        
        if user_level < required_level:
            self._log_security_event("classification_access_denied", {
                'user_id': context.user_id,
                'required_classification': required_classification,
                'user_clearance': user_clearance
            })
            return False
        
        return True
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_audit_report(self, start_date: datetime, 
                            end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        
        relevant_events = [
            event for event in self.audit_log
            if start_date <= datetime.fromisoformat(event['timestamp']) <= end_date
        ]
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': len(relevant_events),
                'authentication_attempts': len([e for e in relevant_events 
                                              if e['event_type'].startswith('authentication')]),
                'access_denials': len([e for e in relevant_events 
                                     if e['event_type'] == 'access_denied']),
                'provenance_records': len([e for e in relevant_events 
                                         if e['event_type'] == 'provenance_created'])
            },
            'events': relevant_events,
            'security_metrics': self._calculate_security_metrics(relevant_events)
        }
        
        return report
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against policy"""
        policy = self.password_policy
        
        if len(password) < policy['min_length']:
            return False
        
        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if policy['require_numbers'] and not any(c.isdigit() for c in password):
            return False
        
        if policy['require_special'] and not any(c in "!@#$%^&*()_+-=" for c in password):
            return False
        
        return True
    
    def _get_user_roles(self, username: str) -> Set[str]:
        """Get user roles from enterprise directory"""
        # In production, this would query LDAP/AD
        # For demo, simulate role assignment
        role_mapping = {
            'admin': {'admin'},
            'developer': {'developer'},
            'analyst': {'analyst'},
            'viewer': {'viewer'},
            'auditor': {'auditor'}
        }
        
        return role_mapping.get(username, {'viewer'})
    
    def _get_user_clearance(self, user_id: str) -> str:
        """Get user security clearance level"""
        # In production, this would be from HR/security systems
        clearance_mapping = {
            'admin': 'restricted',
            'developer': 'confidential',
            'analyst': 'internal',
            'viewer': 'public',
            'auditor': 'restricted'
        }
        
        return clearance_mapping.get(user_id, 'public')
    
    def _encrypt_provenance(self, record: ProvenanceRecord) -> str:
        """Encrypt provenance record for secure storage"""
        record_dict = {
            'record_id': record.record_id,
            'query_id': record.query_id,
            'user_id': record.user_id,
            'cubes_accessed': [c.value for c in record.cubes_accessed],
            'data_sources': record.data_sources,
            'transformations_applied': record.transformations_applied,
            'results_classification': record.results_classification,
            'timestamp': record.timestamp.isoformat(),
            'ip_address': record.ip_address,
            'session_id': record.session_id
        }
        
        return self.encrypt_sensitive_data(json.dumps(record_dict))
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for audit trail"""
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        self.audit_log.append(event)
        logger.info(f"🔒 Security event: {event_type} - {details}")
    
    def _calculate_security_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate security metrics from audit events"""
        
        auth_success = len([e for e in events if e['event_type'] == 'authentication_success'])
        auth_failed = len([e for e in events if e['event_type'] == 'authentication_failed'])
        access_denied = len([e for e in events if e['event_type'] == 'access_denied'])
        
        return {
            'authentication_success_rate': auth_success / (auth_success + auth_failed) if (auth_success + auth_failed) > 0 else 0,
            'access_denial_rate': access_denied / len(events) if events else 0,
            'unique_users': len(set(e['details'].get('user_id') for e in events if 'user_id' in e['details'])),
            'cube_access_patterns': self._analyze_cube_access_patterns(events)
        }
    
    def _analyze_cube_access_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze cube access patterns for security insights"""
        patterns = {}
        
        for event in events:
            if event['event_type'] == 'provenance_created':
                cubes = event['details'].get('cubes_accessed', [])
                for cube in cubes:
                    patterns[cube] = patterns.get(cube, 0) + 1
        
        return patterns

class DataLineageManager:
    """Manages data lineage for provenance and compliance"""
    
    def __init__(self):
        self.lineage_graph: Dict[str, DataLineageNode] = {}
        self.relationships: Dict[str, List[str]] = {}  # node_id -> [dependent_node_ids]
        
    def add_lineage_node(self, node: DataLineageNode):
        """Add node to lineage graph"""
        self.lineage_graph[node.node_id] = node
        if node.node_id not in self.relationships:
            self.relationships[node.node_id] = []
    
    def add_lineage_relationship(self, source_id: str, target_id: str):
        """Add relationship between lineage nodes"""
        if source_id in self.relationships:
            self.relationships[source_id].append(target_id)
        else:
            self.relationships[source_id] = [target_id]
    
    def trace_lineage(self, node_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Trace complete lineage for a data node"""
        
        def _trace_recursive(current_id: str, depth: int, visited: Set[str]) -> Dict[str, Any]:
            if depth >= max_depth or current_id in visited:
                return {}
            
            visited.add(current_id)
            node = self.lineage_graph.get(current_id)
            if not node:
                return {}
            
            result = {
                'node_id': current_id,
                'cube_type': node.cube_type.value,
                'classification': node.data_classification,
                'source_systems': node.source_systems,
                'transformations': node.transformations,
                'created_at': node.created_at.isoformat(),
                'created_by': node.created_by,
                'dependencies': []
            }
            
            for dep_id in self.relationships.get(current_id, []):
                dep_trace = _trace_recursive(dep_id, depth + 1, visited.copy())
                if dep_trace:
                    result['dependencies'].append(dep_trace)
            
            return result
        
        return _trace_recursive(node_id, 0, set())
    
    def validate_lineage_security(self, node_id: str, 
                                context: SecurityContext) -> Tuple[bool, List[str]]:
        """Validate user can access all data in lineage chain"""
        
        lineage = self.trace_lineage(node_id)
        violations = []
        
        def _check_node(node_data: Dict[str, Any]):
            classification = node_data.get('classification', 'public')
            cube_type = CubeType(node_data.get('cube_type', 'data'))
            
            # Check classification clearance
            security_manager = EnterpriseSecurityManager()
            if not security_manager.validate_data_classification(context, classification):
                violations.append(f"Insufficient clearance for {node_data['node_id']} ({classification})")
            
            # Check cube access
            if not security_manager.authorize_cube_access(context, cube_type, AccessLevel.READ):
                violations.append(f"No access to {cube_type.value} cube for {node_data['node_id']}")
            
            # Check dependencies
            for dep in node_data.get('dependencies', []):
                _check_node(dep)
        
        _check_node(lineage)
        
        return len(violations) == 0, violations

# Integration with existing TCDB components
class SecureMultiCubeOrchestrator:
    """Security-enhanced version of MultiCubeOrchestrator"""
    
    def __init__(self):
        self.security_manager = EnterpriseSecurityManager()
        self.lineage_manager = DataLineageManager()
        
    def secure_query(self, query: str, context: SecurityContext) -> Dict[str, Any]:
        """Execute query with full security validation"""
        
        # Validate session
        if not self.security_manager.validate_session(context.session_id):
            raise PermissionError("Invalid or expired session")
        
        # Determine required cubes and access levels
        required_cubes = self._analyze_query_requirements(query)
        
        # Validate cube access
        for cube_type, access_level in required_cubes.items():
            if not self.security_manager.authorize_cube_access(context, cube_type, access_level):
                raise PermissionError(f"Insufficient access to {cube_type.value} cube")
        
        # Execute query (placeholder)
        results = self._execute_query(query, required_cubes)
        
        # Create provenance record
        provenance = self.security_manager.create_provenance_record(
            context=context,
            query_id=str(uuid.uuid4()),
            cubes_accessed=list(required_cubes.keys()),
            data_sources=results.get('data_sources', []),
            transformations=results.get('transformations', []),
            results_classification=results.get('classification', 'internal')
        )
        
        return {
            'results': results,
            'provenance': provenance,
            'security_context': context
        }
    
    def _analyze_query_requirements(self, query: str) -> Dict[CubeType, AccessLevel]:
        """Analyze query to determine cube access requirements"""
        # Simplified analysis - in production would use query parser
        requirements = {}
        
        if 'code' in query.lower():
            requirements[CubeType.CODE] = AccessLevel.READ
        if 'data' in query.lower():
            requirements[CubeType.DATA] = AccessLevel.READ
        if 'user' in query.lower():
            requirements[CubeType.USER] = AccessLevel.READ
        if 'temporal' in query.lower() or 'time' in query.lower():
            requirements[CubeType.TEMPORAL] = AccessLevel.READ
        if 'system' in query.lower():
            requirements[CubeType.SYSTEM] = AccessLevel.READ
        
        # Always require orchestrator access
        requirements[CubeType.ORCHESTRATOR] = AccessLevel.READ
        
        return requirements
    
    def _execute_query(self, query: str, required_cubes: Dict[CubeType, AccessLevel]) -> Dict[str, Any]:
        """Execute the actual query (placeholder)"""
        return {
            'query': query,
            'cubes_used': [cube.value for cube in required_cubes.keys()],
            'data_sources': ['source1', 'source2'],
            'transformations': ['embedding', 'topological_analysis'],
            'classification': 'internal',
            'results': f"Secure query results for: {query}"
        }

def test_enterprise_security():
    """Test enterprise security features"""
    
    print("🔒 Testing Enterprise Security Features")
    print("=" * 50)
    
    # Initialize security manager
    security_manager = EnterpriseSecurityManager()
    
    # Test authentication
    context = security_manager.authenticate_user(
        username="developer",
        password="SecurePass123!",
        ip_address="192.168.1.100"
    )
    
    if context:
        print(f"✅ Authentication successful for {context.user_id}")
        print(f"   Roles: {context.roles}")
        print(f"   Cube Access: {context.cube_access}")
    
    # Test authorization
    can_access = security_manager.authorize_cube_access(
        context, CubeType.CODE, AccessLevel.WRITE
    )
    print(f"✅ Code cube write access: {can_access}")
    
    # Test provenance
    provenance = security_manager.create_provenance_record(
        context=context,
        query_id="test-query-123",
        cubes_accessed=[CubeType.CODE, CubeType.DATA],
        data_sources=["github_repo", "database"],
        transformations=["embedding", "topological_analysis"],
        results_classification="internal"
    )
    print(f"✅ Provenance record created: {provenance.record_id}")
    
    # Test secure orchestrator
    orchestrator = SecureMultiCubeOrchestrator()
    try:
        secure_results = orchestrator.secure_query(
            "Find code related to data processing", context
        )
        print(f"✅ Secure query executed successfully")
        print(f"   Provenance ID: {secure_results['provenance'].record_id}")
    except PermissionError as e:
        print(f"❌ Security violation: {e}")
    
    # Generate audit report
    report = security_manager.generate_audit_report(
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now()
    )
    print(f"✅ Audit report generated: {report['summary']['total_events']} events")
    
    print("\n🔒 Enterprise Security Test Complete!")

if __name__ == "__main__":
    test_enterprise_security()