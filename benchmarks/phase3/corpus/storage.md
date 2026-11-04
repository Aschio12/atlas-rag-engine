# Object storage

Kestrel storage keeps files in buckets. A bucket belongs to one region and one project, and every object in it is addressed by a key.

## Limits

A single bucket can hold up to 5 TB of data. A single object can be up to 5 GB when uploaded in one request and up to 1 TB when uploaded in parts. Bucket names must be unique within a region and may contain lowercase letters, numbers and hyphens.

## Durability and availability

Objects are stored redundantly across three availability zones, which gives a designed durability of 99.999999999 percent. Availability of the storage API is 99.9 percent per month.

## Lifecycle rules

Lifecycle rules move objects to the cold tier after a number of days or delete them automatically. Cold storage costs less but adds a retrieval delay of up to 12 hours. A rule can be limited to a key prefix or to objects with a given tag.

## Versioning and replication

Turn on versioning to keep every version of an object and recover from accidental deletes. Cross-region replication copies new objects to a bucket in another region, usually within 15 minutes.
