# Data residency and privacy

You choose the region where your data lives and it does not leave that region unless you replicate it yourself.

## Regions

Three regions are available: eu-central in Frankfurt, us-east in Virginia and ap-south in Mumbai. The region is chosen when a project is created and cannot be changed later; to move, create a new project and copy the data.

## GDPR

Kestrel acts as a processor of your personal data. A data processing agreement is part of the standard terms and is accepted automatically when you create an organization. A list of sub-processors is published on the trust center and customers are notified 30 days before it changes.

## Exporting and deleting data

Export your data at any time with the CLI command `kestrel export`, which writes a compressed archive to a bucket of your choice. Deleting a project removes its data from live systems immediately and from backups within 35 days.

## Support access

Support staff can access your project only after you grant temporary access, which expires automatically after 24 hours. Every access is written to the audit log.
