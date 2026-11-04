# Compute instances

Instances run your containers. Each instance has a fixed amount of CPU and memory determined by its type.

## Instance types

The k1.small type has 2 vCPU and 4 GB of memory. The k1.medium type has 4 vCPU and 16 GB. The k1.large type has 16 vCPU and 64 GB. Memory-optimized m1 types provide eight times as much memory as vCPUs.

## Autoscaling

Autoscaling adds instances when average CPU utilization stays above a target for five minutes and removes them when it stays below. You choose the minimum and maximum number of instances. Scaling out takes about 90 seconds; scaling in waits for connections to drain for up to 30 seconds.

## Spot instances

Spot instances use spare capacity and cost about 70 percent less than regular instances. They can be reclaimed with a two-minute warning, so use them for batch jobs and stateless workers, never for databases.

## Cold starts

Services configured to scale to zero start in roughly 300 milliseconds when a request arrives after an idle period. Keeping one instance warm removes the cold start completely.
