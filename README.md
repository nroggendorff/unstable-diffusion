# unstable-diffusion

A diffusion trainer that discourages boring output.

## Definitions

| Key | Value |
|-----|-------|
| Basin | The region of latent space holding spokeless images: the average of everything the model has seen, present nowhere in the dataset |
| Spoke | An artistic direction out of the basin. For example, many artists draw noses in a unique way; with the style unspecified, the model should pick one rather than average them |
| Radius | How far an image sits from the basin |
| Angle | Which spoke it committed to. Which style did this seed land in? |
| Orbit | Holding radius while varying angle: staying out of the basin, and landing somewhere different each time |
| Garbage | Random pixels, formless images, anything outside the radius dividing art from spokeless "slop" |
| Adherence | The success metric for finding a point in the manifold that complies with the prompt |
| Variance | Stylistic choices, such as color palette, artstyle/genre, or composition |

Radius and angle are independent, and both can fail on their own. Several images boldly committed to the same unique style have radius but no angle variance.

## Design Philosophies

- Create images that are unique and deliberate, rather than the average of the dataset that doesn't exist at any point in the dataset--if the dataset is a ring of points in image space, a normal trainer would encourage the model to predict somewhere in the center of the ring, not somewhere on the ring (hence orbiting).
- High seed variance should be a biproduct of a well-trained model accompanied by a well-tuned sampler, not a specialized reward term.

The spokes are already in the model. Denoising is trained on a likelihood bound, which is mode-covering: it is punished for assigning near-zero density to a real training image. This is why a small-scope style LoRA, or a very specific prompt work at all, and it means the spokes survive training.

## Mental model

- A random point in image space is just noise.
- A random point in latent space is still noise, but decoded into some semblance of form: shards and blotches of color (when decoded).
- A random point in diffuser space (actually a random point in latent space after a forward pass through the diffuser (including sampling for multiple passes)) is a plausible image with no emergent variance or style--the average basin; you have to steer away from the basin in the conditioning manually to prevent falling in. That's exactly what the sampler should be doing.

We want to create a new manifold that exists after pulling down all of the basins with gravity, making random noise in our space an actually artistically interesting image. The manifold is several islands of interesting images above sea level (the dividing line between art and garbage) in latent space. A random point in that space should be an image an artist could plausibly have made.

## Evaluating this

| Key | Value |
|-----|-------|
| Angular spread | mean pairwise distance between samples in a semantic embedding space. |
| Radius | distance from the basin, measured as displacement from the unconditioned or heavily-averaged centroid. |
| Adherence | similarity to the prompt, as a guard against drifting off-subject rather than as a measure of quality. |

None of these measures whether the result is any *good*. A well-placed detail and a poorly-placed one score identically. Every metric here can confirm that samples differ and that they left the basin; none can confirm they arrived anywhere worth being. That judgement stays human (unfortunately).
