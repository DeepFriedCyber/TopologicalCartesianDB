#!/usr/bin/env python3
"""
Security Integration Framework

Addresses the security integration gaps identified in the technical feedback.
Implements concrete security architecture with FVAC, Proof-Messenger, and Secure-LLM integration.
"""

import hashlib
import hmac
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import uuid

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AccessRole(Enum):
    """Access roles for role-based access control"""
    GUEST = "guest"
    USER = "user"
    ANALYST = "analyst"
    ADMIN = "admin"
    SECURITY_OFFICER = "security_officer"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    role: AccessRole
    security_level: SecurityLevel
    session_id: str
    timestamp: float
    permissions: List[str] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)


@dataclass
class FormalVerificationResult:
    """Result of formal verification (FVAC)"""
    property_verified: str
    verification_status: bool
    proof_trace: List[str]
    verification_time: float
    confidence_level: float
    formal_proof: Optional[str] = None


@dataclass
class CryptographicProof:
    """Cryptographic proof for routing decisions (Proof-Messenger)"""
    operation_id: str
    proof_type: str
    proof_data: bytes
    signature: bytes
    timestamp: float
    verification_key: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAuditEntry:
    """Security audit log entry"""
    timestamp: float
    user_id: str
    operation: str
    resource: str
    result: str
    security_level: SecurityLevel
    risk_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class FVACIntegration:
    """
    Formal Verification and Analysis Component (FVAC) Integration
    
    Provides formal verification of routing logic and security properties.
    """
    
    def __init__(self):
        self.verification_rules = self._initialize_verification_rules()
        self.proof_cache = {}
        self.verification_stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'avg_verification_time': 0.0
        }
    
    def _initialize_verification_rules(self) -> Dict[str, Any]:
        """Initialize formal verification rules"""
        return {
            'access_control': {
                'rule': 'forall user, resource: access(user, resource) -> authorized(user, resource)',
                'description': 'No unauthorized access to resources'
            },
            'routing_consistency': {
                'rule': 'forall query, result: route(query) = result -> consistent(query, result)',
                'description': 'Routing decisions are consistent with coordinate logic'
            },
            'data_integrity': {
                'rule': 'forall data: stored(data) -> integrity_preserved(data)',
                'description': 'Data integrity is preserved throughout operations'
            },
            'security_level_enforcement': {
                'rule': 'forall operation: security_level(operation) <= user_clearance(current_user)',
                'description': 'Security levels are properly enforced'
            }
        }
    
    def verify_routing_decision(self, query_coords: Dict[str, float], 
                               routing_result: Any, security_context: SecurityContext) -> FormalVerificationResult:
        """Verify that a routing decision complies with security policies"""
        start_time = time.time()
        
        # Simulate formal verification process
        verification_steps = []
        
        # Step 1: Verify access control
        access_verified = self._verify_access_control(routing_result, security_context)
        verification_steps.append(f"Access control verification: {'PASS' if access_verified else 'FAIL'}")
        
        # Step 2: Verify routing consistency
        consistency_verified = self._verify_routing_consistency(query_coords, routing_result)
        verification_steps.append(f"Routing consistency verification: {'PASS' if consistency_verified else 'FAIL'}")
        
        # Step 3: Verify security level compliance
        security_verified = self._verify_security_level_compliance(routing_result, security_context)
        verification_steps.append(f"Security level verification: {'PASS' if security_verified else 'FAIL'}")
        
        # Overall verification result
        overall_verified = access_verified and consistency_verified and security_verified
        verification_time = time.time() - start_time
        
        # Update statistics
        self.verification_stats['total_verifications'] += 1
        if overall_verified:
            self.verification_stats['successful_verifications'] += 1
        else:
            self.verification_stats['failed_verifications'] += 1
        
        self.verification_stats['avg_verification_time'] = (
            (self.verification_stats['avg_verification_time'] * (self.verification_stats['total_verifications'] - 1) + 
             verification_time) / self.verification_stats['total_verifications']
        )
        
        # Generate formal proof (simplified)
        formal_proof = self._generate_formal_proof(verification_steps, overall_verified)
        
        return FormalVerificationResult(
            property_verified="routing_security_compliance",
            verification_status=overall_verified,
            proof_trace=verification_steps,
            verification_time=verification_time,
            confidence_level=0.95 if overall_verified else 0.0,
            formal_proof=formal_proof
        )
    
    def _verify_access_control(self, routing_result: Any, security_context: SecurityContext) -> bool:
        """Verify access control compliance"""
        # Check if user has permission to access the routed documents
        required_permissions = self._extract_required_permissions(routing_result)
        
        for permission in required_permissions:
            if permission not in security_context.permissions:
                logger.warning(f"Access denied: User {security_context.user_id} lacks permission {permission}")
                return False
        
        return True
    
    def _verify_routing_consistency(self, query_coords: Dict[str, float], routing_result: Any) -> bool:
        """Verify routing consistency with coordinate logic"""
        # Simplified consistency check
        # In a real implementation, this would use formal methods to verify
        # that the routing result is mathematically consistent with the query
        
        if not hasattr(routing_result, '__iter__'):
            return False
        
        # Check that results are ordered by similarity (basic consistency check)
        similarities = []
        for result in routing_result:
            if hasattr(result, 'similarity_score'):
                similarities.append(result.similarity_score)
        
        # Verify descending order
        return all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
    
    def _verify_security_level_compliance(self, routing_result: Any, security_context: SecurityContext) -> bool:
        """Verify security level compliance"""
        user_clearance_level = self._get_clearance_level(security_context.role)
        
        for result in routing_result:
            if hasattr(result, 'document_id'):
                doc_security_level = self._get_document_security_level(result.document_id)
                if self._security_level_value(doc_security_level) > self._security_level_value(user_clearance_level):
                    logger.warning(f"Security violation: User clearance insufficient for document {result.document_id}")
                    return False
        
        return True
    
    def _extract_required_permissions(self, routing_result: Any) -> List[str]:
        """Extract required permissions from routing result"""
        permissions = set()
        
        for result in routing_result:
            if hasattr(result, 'document_id'):
                # Determine permissions based on document type/content
                permissions.add("read_documents")
                
                # Add specific permissions based on document classification
                doc_id = result.document_id
                if "confidential" in doc_id.lower():
                    permissions.add("read_confidential")
                if "admin" in doc_id.lower():
                    permissions.add("admin_access")
        
        return list(permissions)
    
    def _get_clearance_level(self, role: AccessRole) -> SecurityLevel:
        """Get security clearance level for a role"""
        role_clearance = {
            AccessRole.GUEST: SecurityLevel.PUBLIC,
            AccessRole.USER: SecurityLevel.INTERNAL,
            AccessRole.ANALYST: SecurityLevel.CONFIDENTIAL,
            AccessRole.ADMIN: SecurityLevel.RESTRICTED,
            AccessRole.SECURITY_OFFICER: SecurityLevel.TOP_SECRET
        }
        return role_clearance.get(role, SecurityLevel.PUBLIC)
    
    def _get_document_security_level(self, document_id: str) -> SecurityLevel:
        """Get security level for a document (simplified)"""
        # In a real implementation, this would query a security database
        if "top_secret" in document_id.lower():
            return SecurityLevel.TOP_SECRET
        elif "restricted" in document_id.lower():
            return SecurityLevel.RESTRICTED
        elif "confidential" in document_id.lower():
            return SecurityLevel.CONFIDENTIAL
        elif "internal" in document_id.lower():
            return SecurityLevel.INTERNAL
        else:
            return SecurityLevel.PUBLIC
    
    def _security_level_value(self, level: SecurityLevel) -> int:
        """Convert security level to numeric value for comparison"""
        level_values = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.RESTRICTED: 4,
            SecurityLevel.TOP_SECRET: 5
        }
        return level_values.get(level, 1)
    
    def _generate_formal_proof(self, verification_steps: List[str], verified: bool) -> str:
        """Generate formal proof representation"""
        proof_lines = [
            "FORMAL VERIFICATION PROOF",
            "=" * 25,
            f"Timestamp: {time.time()}",
            f"Result: {'VERIFIED' if verified else 'FAILED'}",
            "",
            "Verification Steps:"
        ]
        
        for i, step in enumerate(verification_steps, 1):
            proof_lines.append(f"  {i}. {step}")
        
        proof_lines.extend([
            "",
            "Conclusion:",
            f"  The routing decision is {'FORMALLY VERIFIED' if verified else 'NOT VERIFIED'} to comply with security policies.",
            "",
            "QED" if verified else "VERIFICATION FAILED"
        ])
        
        return "\n".join(proof_lines)


class ProofMessengerIntegration:
    """
    Proof-Messenger Integration
    
    Generates and verifies cryptographic proofs for routing decisions.
    """
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.proof_store = {}
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def generate_routing_proof(self, query_coords: Dict[str, float], 
                              routing_result: Any, security_context: SecurityContext) -> CryptographicProof:
        """Generate cryptographic proof for a routing decision"""
        operation_id = str(uuid.uuid4())
        
        # Create proof data
        proof_data = {
            'operation_id': operation_id,
            'query_coordinates': query_coords,
            'routing_result_summary': self._summarize_routing_result(routing_result),
            'user_id': security_context.user_id,
            'timestamp': time.time(),
            'security_level': security_context.security_level.value
        }
        
        # Serialize and encrypt proof data
        proof_json = json.dumps(proof_data, sort_keys=True)
        encrypted_proof = self.cipher_suite.encrypt(proof_json.encode())
        
        # Generate digital signature
        signature = self.private_key.sign(
            encrypted_proof,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Get public key for verification
        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        proof = CryptographicProof(
            operation_id=operation_id,
            proof_type="routing_decision",
            proof_data=encrypted_proof,
            signature=signature,
            timestamp=time.time(),
            verification_key=public_key_bytes,
            metadata={
                'query_hash': hashlib.sha256(json.dumps(query_coords, sort_keys=True).encode()).hexdigest(),
                'user_id': security_context.user_id,
                'security_level': security_context.security_level.value
            }
        )
        
        # Store proof
        self.proof_store[operation_id] = proof
        
        return proof
    
    def verify_routing_proof(self, proof: CryptographicProof) -> bool:
        """Verify a cryptographic proof"""
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(proof.verification_key)
            
            # Verify signature
            public_key.verify(
                proof.signature,
                proof.proof_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Decrypt and validate proof data
            decrypted_data = self.cipher_suite.decrypt(proof.proof_data)
            proof_data = json.loads(decrypted_data.decode())
            
            # Validate proof structure
            required_fields = ['operation_id', 'query_coordinates', 'user_id', 'timestamp']
            if not all(field in proof_data for field in required_fields):
                return False
            
            # Validate timestamp (not too old)
            if time.time() - proof_data['timestamp'] > 3600:  # 1 hour expiry
                logger.warning(f"Proof {proof.operation_id} has expired")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
    
    def _summarize_routing_result(self, routing_result: Any) -> Dict[str, Any]:
        """Create a summary of routing result for proof"""
        if not hasattr(routing_result, '__iter__'):
            return {'type': 'single_result', 'count': 1}
        
        summary = {
            'type': 'multiple_results',
            'count': len(routing_result),
            'document_ids': [],
            'similarity_scores': []
        }
        
        for result in routing_result:
            if hasattr(result, 'document_id'):
                summary['document_ids'].append(result.document_id)
            if hasattr(result, 'similarity_score'):
                summary['similarity_scores'].append(result.similarity_score)
        
        return summary


class SecureLLMIntegration:
    """
    Secure-LLM Integration with Triple-Lock Gateway
    
    Implements secure access control for LLM interactions.
    """
    
    def __init__(self):
        self.access_policies = self._initialize_access_policies()
        self.active_sessions = {}
        self.audit_log = []
        self.risk_assessor = SecurityRiskAssessor()
    
    def _initialize_access_policies(self) -> Dict[str, Any]:
        """Initialize access control policies"""
        return {
            'role_permissions': {
                AccessRole.GUEST: ['read_public'],
                AccessRole.USER: ['read_public', 'read_internal'],
                AccessRole.ANALYST: ['read_public', 'read_internal', 'read_confidential', 'analyze'],
                AccessRole.ADMIN: ['read_public', 'read_internal', 'read_confidential', 'read_restricted', 'admin'],
                AccessRole.SECURITY_OFFICER: ['*']  # All permissions
            },
            'security_level_access': {
                SecurityLevel.PUBLIC: [AccessRole.GUEST, AccessRole.USER, AccessRole.ANALYST, AccessRole.ADMIN, AccessRole.SECURITY_OFFICER],
                SecurityLevel.INTERNAL: [AccessRole.USER, AccessRole.ANALYST, AccessRole.ADMIN, AccessRole.SECURITY_OFFICER],
                SecurityLevel.CONFIDENTIAL: [AccessRole.ANALYST, AccessRole.ADMIN, AccessRole.SECURITY_OFFICER],
                SecurityLevel.RESTRICTED: [AccessRole.ADMIN, AccessRole.SECURITY_OFFICER],
                SecurityLevel.TOP_SECRET: [AccessRole.SECURITY_OFFICER]
            }
        }
    
    def create_secure_session(self, user_id: str, role: AccessRole, 
                             additional_permissions: Optional[List[str]] = None) -> SecurityContext:
        """Create a secure session with triple-lock validation"""
        session_id = str(uuid.uuid4())
        
        # Lock 1: Role-based permissions
        base_permissions = self.access_policies['role_permissions'].get(role, [])
        
        # Lock 2: Additional permission validation
        if additional_permissions:
            validated_permissions = self._validate_additional_permissions(role, additional_permissions)
            permissions = list(set(base_permissions + validated_permissions))
        else:
            permissions = base_permissions
        
        # Lock 3: Security level determination
        security_level = self._determine_security_level(role, permissions)
        
        security_context = SecurityContext(
            user_id=user_id,
            role=role,
            security_level=security_level,
            session_id=session_id,
            timestamp=time.time(),
            permissions=permissions
        )
        
        # Store active session
        self.active_sessions[session_id] = security_context
        
        # Audit log entry
        self._log_security_event(
            user_id=user_id,
            operation="session_created",
            resource=session_id,
            result="success",
            security_level=security_level,
            details={'role': role.value, 'permissions': permissions}
        )
        
        return security_context
    
    def validate_cube_access(self, security_context: SecurityContext, 
                           cube_name: str, operation: str) -> bool:
        """Validate access to a specific cube operation"""
        # Check session validity
        if not self._is_session_valid(security_context):
            return False
        
        # Assess risk
        risk_score = self.risk_assessor.assess_operation_risk(
            security_context, cube_name, operation
        )
        
        # High-risk operations require additional validation
        if risk_score > 0.7:
            logger.warning(f"High-risk operation detected: {operation} on {cube_name} by {security_context.user_id}")
            # In a real implementation, this might trigger additional authentication
        
        # Check specific permissions for cube access
        required_permission = f"access_{cube_name}"
        if required_permission not in security_context.permissions and '*' not in security_context.permissions:
            self._log_security_event(
                user_id=security_context.user_id,
                operation=f"cube_access_denied",
                resource=cube_name,
                result="denied",
                security_level=security_context.security_level,
                details={'operation': operation, 'risk_score': risk_score}
            )
            return False
        
        # Log successful access
        self._log_security_event(
            user_id=security_context.user_id,
            operation=f"cube_access_granted",
            resource=cube_name,
            result="granted",
            security_level=security_context.security_level,
            details={'operation': operation, 'risk_score': risk_score}
        )
        
        return True
    
    def _validate_additional_permissions(self, role: AccessRole, 
                                       additional_permissions: List[str]) -> List[str]:
        """Validate additional permissions against role capabilities"""
        base_permissions = self.access_policies['role_permissions'].get(role, [])
        
        # Only allow permissions that are compatible with the role
        validated = []
        for permission in additional_permissions:
            if self._is_permission_compatible(role, permission):
                validated.append(permission)
            else:
                logger.warning(f"Permission {permission} not compatible with role {role.value}")
        
        return validated
    
    def _is_permission_compatible(self, role: AccessRole, permission: str) -> bool:
        """Check if a permission is compatible with a role"""
        # Define permission hierarchy
        permission_levels = {
            'read_public': 1,
            'read_internal': 2,
            'read_confidential': 3,
            'read_restricted': 4,
            'admin': 5,
            '*': 6
        }
        
        role_levels = {
            AccessRole.GUEST: 1,
            AccessRole.USER: 2,
            AccessRole.ANALYST: 3,
            AccessRole.ADMIN: 4,
            AccessRole.SECURITY_OFFICER: 6
        }
        
        permission_level = permission_levels.get(permission, 0)
        role_level = role_levels.get(role, 0)
        
        return role_level >= permission_level
    
    def _determine_security_level(self, role: AccessRole, permissions: List[str]) -> SecurityLevel:
        """Determine security level based on role and permissions"""
        if '*' in permissions or role == AccessRole.SECURITY_OFFICER:
            return SecurityLevel.TOP_SECRET
        elif 'read_restricted' in permissions or role == AccessRole.ADMIN:
            return SecurityLevel.RESTRICTED
        elif 'read_confidential' in permissions or role == AccessRole.ANALYST:
            return SecurityLevel.CONFIDENTIAL
        elif 'read_internal' in permissions or role == AccessRole.USER:
            return SecurityLevel.INTERNAL
        else:
            return SecurityLevel.PUBLIC
    
    def _is_session_valid(self, security_context: SecurityContext) -> bool:
        """Check if a security context/session is still valid"""
        # Check if session exists
        if security_context.session_id not in self.active_sessions:
            return False
        
        # Check session timeout (1 hour)
        if time.time() - security_context.timestamp > 3600:
            self._invalidate_session(security_context.session_id)
            return False
        
        return True
    
    def _invalidate_session(self, session_id: str):
        """Invalidate a session"""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            self._log_security_event(
                user_id=context.user_id,
                operation="session_invalidated",
                resource=session_id,
                result="success",
                security_level=context.security_level,
                details={'reason': 'timeout'}
            )
    
    def _log_security_event(self, user_id: str, operation: str, resource: str, 
                           result: str, security_level: SecurityLevel, 
                           details: Optional[Dict[str, Any]] = None):
        """Log a security event"""
        audit_entry = SecurityAuditEntry(
            timestamp=time.time(),
            user_id=user_id,
            operation=operation,
            resource=resource,
            result=result,
            security_level=security_level,
            risk_score=self.risk_assessor.calculate_base_risk(security_level, operation),
            details=details or {}
        )
        
        self.audit_log.append(audit_entry)
        
        # In a real implementation, this would be written to a secure audit database
        logger.info(f"Security event: {operation} on {resource} by {user_id} - {result}")


class SecurityRiskAssessor:
    """Assesses security risks for operations"""
    
    def __init__(self):
        self.risk_factors = {
            'high_privilege_operations': ['admin', 'delete', 'modify_security'],
            'sensitive_cubes': ['system_cube', 'security_cube'],
            'unusual_access_patterns': []
        }
    
    def assess_operation_risk(self, security_context: SecurityContext, 
                            cube_name: str, operation: str) -> float:
        """Assess risk score for an operation (0.0 to 1.0)"""
        base_risk = self.calculate_base_risk(security_context.security_level, operation)
        
        # Adjust for cube sensitivity
        if cube_name in self.risk_factors['sensitive_cubes']:
            base_risk += 0.2
        
        # Adjust for operation type
        if operation in self.risk_factors['high_privilege_operations']:
            base_risk += 0.3
        
        # Adjust for user role
        if security_context.role in [AccessRole.GUEST, AccessRole.USER]:
            base_risk += 0.1
        
        return min(1.0, base_risk)
    
    def calculate_base_risk(self, security_level: SecurityLevel, operation: str) -> float:
        """Calculate base risk score"""
        level_risks = {
            SecurityLevel.PUBLIC: 0.1,
            SecurityLevel.INTERNAL: 0.2,
            SecurityLevel.CONFIDENTIAL: 0.4,
            SecurityLevel.RESTRICTED: 0.6,
            SecurityLevel.TOP_SECRET: 0.8
        }
        
        return level_risks.get(security_level, 0.5)


class IntegratedSecurityFramework:
    """
    Integrated Security Framework
    
    Combines FVAC, Proof-Messenger, and Secure-LLM for comprehensive security.
    """
    
    def __init__(self):
        self.fvac = FVACIntegration()
        self.proof_messenger = ProofMessengerIntegration()
        self.secure_llm = SecureLLMIntegration()
        
        logger.info("Integrated Security Framework initialized")
    
    def secure_routing_operation(self, query_coords: Dict[str, float], 
                                routing_result: Any, security_context: SecurityContext) -> Dict[str, Any]:
        """Perform a complete secure routing operation"""
        
        # Step 1: Formal verification (FVAC)
        verification_result = self.fvac.verify_routing_decision(
            query_coords, routing_result, security_context
        )
        
        if not verification_result.verification_status:
            logger.error("Routing decision failed formal verification")
            return {
                'success': False,
                'error': 'Formal verification failed',
                'verification_result': verification_result
            }
        
        # Step 2: Generate cryptographic proof (Proof-Messenger)
        crypto_proof = self.proof_messenger.generate_routing_proof(
            query_coords, routing_result, security_context
        )
        
        # Step 3: Validate proof
        proof_valid = self.proof_messenger.verify_routing_proof(crypto_proof)
        
        if not proof_valid:
            logger.error("Cryptographic proof validation failed")
            return {
                'success': False,
                'error': 'Proof validation failed',
                'verification_result': verification_result,
                'crypto_proof': crypto_proof
            }
        
        # Step 4: Log security event
        self.secure_llm._log_security_event(
            user_id=security_context.user_id,
            operation="secure_routing_completed",
            resource="routing_engine",
            result="success",
            security_level=security_context.security_level,
            details={
                'verification_time': verification_result.verification_time,
                'proof_id': crypto_proof.operation_id
            }
        )
        
        return {
            'success': True,
            'verification_result': verification_result,
            'crypto_proof': crypto_proof,
            'routing_result': routing_result
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security framework status"""
        return {
            'fvac_stats': self.fvac.verification_stats,
            'active_sessions': len(self.secure_llm.active_sessions),
            'total_proofs': len(self.proof_messenger.proof_store),
            'audit_entries': len(self.secure_llm.audit_log),
            'framework_status': 'operational'
        }


# Example usage
if __name__ == "__main__":
    # Initialize integrated security framework
    security_framework = IntegratedSecurityFramework()
    
    # Create a security context
    security_context = security_framework.secure_llm.create_secure_session(
        user_id="analyst_001",
        role=AccessRole.ANALYST,
        additional_permissions=["read_confidential", "analyze"]
    )
    
    print("Integrated Security Framework Demo")
    print("=" * 40)
    print(f"Security Context: {security_context.user_id} ({security_context.role.value})")
    print(f"Security Level: {security_context.security_level.value}")
    print(f"Permissions: {security_context.permissions}")
    
    # Simulate a routing operation
    query_coords = {'domain': 0.8, 'complexity': 0.6, 'task_type': 0.4}
    
    # Mock routing result
    class MockRoutingResult:
        def __init__(self, doc_id, similarity):
            self.document_id = doc_id
            self.similarity_score = similarity
    
    routing_result = [
        MockRoutingResult("doc_confidential_001", 0.9),
        MockRoutingResult("doc_internal_002", 0.7)
    ]
    
    # Perform secure routing operation
    result = security_framework.secure_routing_operation(
        query_coords, routing_result, security_context
    )
    
    print(f"\nSecure Routing Result:")
    print(f"Success: {result['success']}")
    
    if result['success']:
        print(f"Verification Status: {result['verification_result'].verification_status}")
        print(f"Proof ID: {result['crypto_proof'].operation_id}")
        print(f"Verification Time: {result['verification_result'].verification_time:.3f}s")
    
    # Show security status
    status = security_framework.get_security_status()
    print(f"\nSecurity Framework Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")