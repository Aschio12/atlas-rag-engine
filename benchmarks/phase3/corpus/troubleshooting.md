# Troubleshooting

This page lists the errors we see most often and how to fix them.

## ERR-1007 quota exceeded

The organization reached a resource quota, such as the maximum number of instances or buckets. Delete unused resources or ask support to raise the quota. Quotas are shown under Settings and then Limits.

## ERR-2210 deployment failed

The container image could not start. Open the deployment logs: the most common causes are a missing environment variable, a port other than the one declared in the configuration and a start command that exits immediately.

## TIMEOUT-3 request timed out

The request exceeded the 30 second gateway timeout. Move long work to a background job and return a job identifier instead, or stream the response in chunks.

## Slow uploads

Large files should be uploaded in parts of 64 MB with four parallel connections. Uploads from far away are faster when you use the region closest to your users.

## Contacting support

Include the request identifier from the X-Request-Id response header. With it, support can find your request in seconds. Pro customers get a first response within 4 hours and Enterprise customers within 1 hour.
