# Security and compliance

Security is shared between Kestrel and its customers. This page lists what we do and what remains your responsibility.

## Encryption

Data at rest is encrypted with AES-256 and data in transit with TLS 1.3. Customers on the Enterprise plan can bring their own encryption keys and revoke them at any time, which makes the data unreadable.

## Certifications

Kestrel Cloud holds a SOC 2 Type II report and ISO 27001 certification. Reports are available under NDA from the trust center. We run an external penetration test every year.

## Audit logs

Audit logs record sign-ins, permission changes and API calls that modify resources. They are retained for 400 days and can be streamed to your own storage bucket.

## Access control

Roles are Owner, Admin, Developer and Viewer. Use scoped API keys and short-lived tokens wherever you can, and require multi-factor authentication for every human user.

## Reporting a vulnerability

Send reports to the security team through the disclosure form. We acknowledge reports within two business days and do not take legal action against good-faith research.
