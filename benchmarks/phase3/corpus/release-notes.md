# Release notes

## Version 2.4.0 (March 2026)

Added scheduled scaling, so that instance counts can follow a weekly calendar. Added private endpoints for object storage. The default log retention for the Pro plan increased from 14 days to 30 days.

## Version 2.3.1 (February 2026)

Fixed a bug where refresh tokens were not rotated after a failed refresh. Fixed the display of the Retry-After header in the dashboard request inspector.

## Version 2.3.0 (January 2026)

Introduced the ap-south region in Mumbai. Added hardware security key support for multi-factor authentication. Removed the legacy v1 authentication endpoint, which returned error AUTH-4001 for all requests after the removal date.

## Version 2.2.0 (November 2025)

Added spot instances with a two-minute reclaim warning. Raised the maximum object size for multipart uploads from 500 GB to 1 TB.
