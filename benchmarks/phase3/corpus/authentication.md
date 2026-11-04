# Authentication

Every request to the Kestrel API must be authenticated. Two methods are supported: API keys for servers and OAuth 2.0 for applications that act on behalf of users.

## API keys

Send the key in the Authorization header as a bearer token. Keys do not expire by themselves, but we recommend rotating them every 90 days. Rotation creates a new key and keeps the old one valid for 24 hours so that deployments can switch over without downtime.

## OAuth 2.0

Applications use the authorization code flow with PKCE. Access tokens are valid for 3600 seconds. Refresh tokens are valid for 30 days and are rotated every time they are used, so a refresh token can be used only once.

## Multi-factor authentication

Organization owners can require MFA for all members. Members then sign in with a time-based one-time password from an authenticator app or with a hardware security key.

## Authentication errors

Error AUTH-4012 means that the access token has expired; request a new token with your refresh token. Error AUTH-4013 means that the API key was revoked. Error AUTH-4031 means the key is valid but lacks the scope required for the operation.
