# Rate limits

Rate limits protect the platform and keep latency predictable for everyone. Limits apply per organization and are counted over a sliding window of one minute.

## Limits by plan

The Free plan allows 60 requests per minute. The Pro plan allows 600 requests per minute. The Enterprise plan allows 6000 requests per minute, and higher limits can be agreed in the contract. Batch endpoints count as one request regardless of how many items they contain.

## Response headers

Every response includes the headers X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset. The reset header contains the number of seconds until the window allows new requests.

## When the limit is exceeded

The API answers with HTTP status 429 and a Retry-After header. Clients should wait for the number of seconds in Retry-After and then retry. We recommend exponential backoff with random jitter, starting at one second and doubling up to a maximum of 60 seconds.

## Requesting an increase

Pro customers can request a temporary increase from the support portal at least two business days before a planned launch or load test.
