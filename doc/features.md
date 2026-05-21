### Runpod remote training

It would be crucial for transfereablitlity to have a remote training option. This would allow users to train their models on powerful cloud GPUs without needing to set up complex environments locally. Runpod is a great option for this, as it provides easy access to GPU resources and can be integrated into our workflow for seamless model training and deployment.

#### Option 1: self contained docker image
We could create a self-contained docker image that includes all the necessary dependencies for training our models. This image could be pushed to a container registry and then pulled and run on Runpod for training.


#### Option 2: remote training api
Allow for remote training by exposing an API endpoint that accepts training data and configuration parameters. This API would then trigger the training process on Runpod, allowing users to train their models without needing to manage the underlying infrastructure. 
A simple internal queuing should exist. 


### early stopping and model checkpointing ✅ done (2026-05-20)

Implemented in-process. See `CLAUDE.md` → **Early stopping** and the design
spec at `docs/superpowers/specs/2026-05-20-early-stopping-design.md`.