# Networking

Every project has a private network called a VPC. Instances in the same VPC talk to each other over private addresses and never cross the public internet.

## Public access and firewall rules

By default an instance has no public address. Attach a load balancer to expose a service on ports 80 and 443. Firewall rules allow or deny traffic by protocol, port and source range, and rules are evaluated from the lowest priority number to the highest.

## IP allowlists

Add an IP allowlist to an API key to accept requests only from your office or data center ranges. Requests from other addresses are rejected with status 403.

## VPC peering and private endpoints

Peering connects two VPCs, even across regions, with no overlap in their address ranges. Private endpoints let you reach Kestrel storage and databases from inside a VPC without a public route.

## Network errors

Error NET-5031 means the destination port is blocked by a firewall rule; check the rule priorities. Error NET-5044 means that two peered VPCs have overlapping address ranges and the peering cannot be created.
