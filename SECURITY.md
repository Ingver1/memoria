# Security

See [architecture documentation](./docs/architecture.md) for an overview of the
system architecture, data flow, and threat model.

Memoria uses [SQLCipher](https://www.zetetic.net/sqlcipher/) to provide
transparent encryption for SQLite databases. When
`security.encrypt_at_rest` is enabled, the SQLite driver applies
`PRAGMA key` on every connection before any other commands execute,
ensuring that the database cannot be read without the correct key.

### Key Rotation

The `SQLiteMemoryStore` exposes `rotate_key(new_key)` which internally
uses `PRAGMA rekey` to re-encrypt the database with a new password.
Existing connections are closed so subsequent operations require the new
key. After rotating, attempts to access the database with the old key
will fail with `sqlite3.DatabaseError`.

## Key & Secret Management
- Store keys only in environment variables or secret managers; never commit to VCS.
- Rotate keys on compromise or at least every **90 days**.
- After `rotate_key()`, revoke any old backups containing previous keys.

## Reporting a Vulnerability
Please report security issues privately via **GitHub Private Vulnerability Reporting** (Security → "Report a vulnerability")
or email **https://kayel.20221967@gmail.com**.  
Do **not** open public issues with sensitive details.

## Supported Versions
We actively support the latest minor release.  
Security fixes are backported to the **latest** patch only.

## Scope
- Code in this repo and default configurations
- Encryption-at-rest (SQLCipher) configuration
- Memory store adapters maintained here

### Out of Scope
- Third-party deployments and forks we don’t operate
- User custom code or external plugins
- Social engineering

## Responsible Disclosure / Safe Harbor
Non-destructive, good-faith research is welcome.  
Please avoid data exfiltration, service disruption, and privacy violations.

## Security Requirements Status

| Requirement | Status |
|-------------|--------|
| V1: [Architecture, Design and Threat Modeling](./docs/architecture.md) | Done |
| V2: Authentication | Pending |
| V3: Session Management | Pending |
| V4: Access Control | Pending |
| V5: Validation, Sanitization and Encoding | Pending |
| V6: Stored Cryptography | Pending |
| V7: Error Handling and Logging | Pending |
| V8: Data Protection | Pending |
| V9: Communications | Pending |
| V10: Malicious Code | Pending |
| V11: Business Logic | Pending |
| V12: Files and Resources | Pending |
| V13: API and Web Services | Pending |
