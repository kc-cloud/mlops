## Layer 7: GenAI RAG Application Security

### Threat: Prompt injection, PII/PHI leakage, jailbreaking, model abuse

### Key Security Controls

1. **Frontend**: HTTPS/TLS 1.2+, CSP headers, input sanitization, secure cookies
2. **API Gateway + WAF**: OWASP rules, rate limits, prompt injection blocks
3. **Input Guardrails**: Block jailbreaks, scrub PII/PHI, validate queries
4. **Output Guardrails**: Redact PII, detect hallucinations, verify citations
5. **SageMaker Endpoint**: Private VPC, IAM auth, TLS encryption, logging
6. **RAG Knowledge Base**: Encrypt data, validate sources, version control
7. **Monitoring**: CloudWatch alarms, guardrail metrics, anomaly detection
8. **Compliance**: 95%+ controls, quarterly pentests, HIPAA/GDPR/SOC2 certs
