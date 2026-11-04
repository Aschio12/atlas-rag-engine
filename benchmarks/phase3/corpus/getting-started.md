# Getting started with Kestrel Cloud

Kestrel Cloud is a hosted platform for storage, compute and networking. This guide takes you from a new account to your first deployed service.

## Create an account

Sign up with a work email address. Verify the address, then create an organization. The organization owns every project, bucket and instance you create later, and its owner can invite teammates with the Admin, Developer or Viewer role.

## Create an API key

Open Settings and choose API keys, then press Create key. Copy the secret straight away: it is shown only once. Keys start with the prefix kc_live_ and can be scoped to a single project.

## Install the command line tool

Install the CLI with the package manager of your platform and run `kestrel login`. The command opens a browser window and stores a token on your machine. Run `kestrel whoami` to confirm which organization you are signed in to.

## Deploy your first service

Create a project, then run `kestrel deploy` inside your application folder. The first deployment builds a container image, uploads it and starts one instance of type k1.small. The service receives a public address ending in kestrel.example within about two minutes.
