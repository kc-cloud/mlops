## Layer 7: GenAI RAG Application Security - Key Controls

1. **Frontend Security**: Enforce HTTPS/TLS 1.2+, implement CSP headers, sanitize all inputs, use secure session management (HttpOnly, Secure cookies), and mask PII/PHI in UI displays

2. **API Gateway + WAF**: Enable OWASP Top 10 rules, implement rate limiting (100 req/5min per IP), block prompt injection patterns ("Ignore previous instructions", "DAN mode"), enforce request size limits (10KB max), and enable geo-blocking

3. **Lambda Input Guardrails**: Detect and block prompt injection/jailbreaking attempts, scrub PII/PHI using regex patterns (SSN, email, medical IDs), limit query length (2000 chars/500 tokens), validate semantic relevance, and implement least-privilege IAM roles

4. **Lambda Output Guardrails**: Redact PII/PHI in responses, implement hallucination detection with confidence scores, filter toxic content using AWS Comprehend, verify citations against RAG corpus, and enforce response token limits (500 max)

5. **SageMaker Endpoint Security**: Use private VPC endpoints only, enforce IAM-based authorization (Lambda role only), enable TLS encryption in transit, implement request/response logging to CloudWatch (90-day retention), and monitor for adversarial input patterns

6. **RAG Knowledge Base Security**: Encrypt vector database and S3 storage with KMS, validate all document sources before indexing, implement access controls on embeddings, enable immutable versioning with checksums, and log all document retrievals for audit trails

7. **Monitoring & Incident Response**: Set CloudWatch alarms for WAF blocks (>10/min), track guardrail metrics (input rejections %, PII detections, jailbreak attempts), enable GuardDuty for anomaly detection, maintain 90-day CloudWatch logs + 7-year S3 archives, and implement automated incident playbooks

8. **Compliance & Testing**: Achieve 95%+ control compliance before production, conduct quarterly penetration testing for prompt injection/jailbreaking/PII leakage, maintain HIPAA/GDPR/SOC2 certifications, and validate all security controls meet PCI-DSS requirements for sensitive data handling