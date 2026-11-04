# Monitoring and alerts

Kestrel collects metrics, logs and traces for every service without extra setup.

## Metrics and dashboards

CPU, memory, request rate, error rate and latency are recorded every 10 seconds. Metrics are kept at full resolution for 15 days and downsampled to one-minute resolution for 13 months. Dashboards can be shared with read-only links.

## Logs

Standard output and standard error of your containers are collected automatically. Logs are retained for 30 days on the Free and Pro plans and for 90 days on Enterprise. Search supports exact phrases and field filters.

## Alerts

An alert watches a metric and notifies a channel when a threshold is crossed for a given time. Channels include email, chat webhooks and the on-call pager integration. Use alert muting during planned maintenance.

## Service level objective

The platform control plane targets an availability of 99.95 percent per month. The public status page shows incidents in real time and lists the root cause analysis within five business days.
